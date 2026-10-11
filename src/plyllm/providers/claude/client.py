from __future__ import annotations

import json
import re
from typing import Any, Literal, cast

from anthropic import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncAnthropic, transform_schema
from anthropic.types import Message, MessageParam, TextBlockParam, Usage
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel

from plyllm.batch import BatchJob
from plyllm.client import LLMClient
from plyllm.errors import FailureKind, LLMCallError, retryable_status, status_kind
from plyllm.llm_config import LLMConfig
from plyllm.message import LLMMessage, MessageRole
from plyllm.options import LLMCallOptions
from plyllm.providers.claude.models import ClaudeLLMModel, claude_models
from plyllm.providers.providers import Provider
from plyllm.response import LLMResponse
from plyllm.tokens import LLMTokens

AnthropicRoles = Literal["user", "assistant", "system"]

# APITimeoutError subclasses APIConnectionError and RateLimitError subclasses APIStatusError, so these two
# cover every failure the SDK raises for a request that was sent
_RETRY_ERRORS = (APIConnectionError, APIStatusError)
# the messages array only carries the conversation; the system prompt is a separate request field
_ROLE_MAP: dict[MessageRole, AnthropicRoles] = {"user": "user", "assistant": "assistant", "system": "user"}
# "up to 100,000 Message requests or 256 MB in size, whichever is reached first"
# (https://platform.claude.com/docs/en/build-with-claude/batch-processing)
_MAX_BATCH_REQUESTS = 100_000
_MAX_BATCH_BYTES = 256 * 1024 * 1024
_CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
# the error of a request that failed inside a batch, by its type; the rest are the request's or the account's fault
_BATCH_ERROR_KINDS = {"rate_limit_error": FailureKind.RATE_LIMIT, "timeout_error": FailureKind.TIMEOUT}


def _thinking_summaries(message: Message) -> list[str]:
    return [block.thinking for block in message.content if block.type == "thinking" and block.thinking]


def _output_text(message: Message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def _total_tokens(message: Message) -> int:
    # what the rate gate charges against a tokens_per_minute quota
    usage = message.usage
    return usage.input_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0) + usage.output_tokens


def message_tokens(usage: Usage) -> LLMTokens:
    # usage.input_tokens counts only the uncached prefix; cache reads and writes are reported separately
    cached_tokens = usage.cache_read_input_tokens or 0
    cache_write_tokens = usage.cache_creation_input_tokens or 0
    return LLMTokens(
        input_tokens=usage.input_tokens + cached_tokens + cache_write_tokens,
        cached_input_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.output_tokens_details.thinking_tokens if usage.output_tokens_details is not None else 0,
    )


class ClaudeLLMClient(LLMClient[ClaudeLLMModel, Message]):
    provider_key = Provider.CLAUDE
    _retry_errors = _RETRY_ERRORS
    supports_batch = True
    max_batch_requests = _MAX_BATCH_REQUESTS
    max_batch_bytes = _MAX_BATCH_BYTES

    def __init__(self, client: AsyncAnthropic, concurrency: int | None = None) -> None:
        super().__init__(claude_models(), [], concurrency)
        self._client = client

    @classmethod
    def build(cls, config: LLMConfig) -> ClaudeLLMClient | None:
        if config.claude is None:
            return None
        client = AsyncAnthropic(api_key=config.claude.api_key, timeout=config.claude.timeout)
        return cls(client, config.default_concurrency)

    def _should_retry_on_error(self, error: Exception) -> bool:
        if isinstance(error, APIConnectionError):
            return True  # never reached the API, or timed out on the way
        return isinstance(error, APIStatusError) and retryable_status(error.status_code)

    def error_kind(self, error: Exception) -> FailureKind:
        if isinstance(error, APITimeoutError):
            return FailureKind.TIMEOUT  # before APIConnectionError, which it subclasses
        if isinstance(error, APIConnectionError):
            return FailureKind.CONNECTION
        if isinstance(error, APIStatusError):
            return status_kind(error.status_code)
        # e.g. APIResponseValidationError: the API answered, with something the SDK could not read
        return FailureKind.PROVIDER if isinstance(error, APIError) else FailureKind.OTHER

    async def aclose(self) -> None:
        await self._client.close()

    async def served_models(self) -> set[str]:
        return {model.id async for model in self._client.models.list()}

    def _request(self, model: ClaudeLLMModel, system: LLMMessage, messages: list[LLMMessage], options: LLMCallOptions, output_schema: type[BaseModel] | None) -> dict[str, Any]:
        # the request body, apart from the call, so a batch request carries exactly what generate() sends
        params = model.extract_params(options)
        # the breakpoint sits on the system prompt, so the varying turns stay outside the cached prefix.
        # TODO: batches use this 5-minute cache too, so both transports send the same body. The batch docs
        # recommend the 1-hour TTL, since a batch spreads requests over a longer time and hits a 5-minute
        # cache less often; but a 1-hour write costs more. Decide once the paid check shows the batch hit rate.
        system_blocks: list[TextBlockParam] = [{"type": "text", "text": system.content, "cache_control": {"type": "ephemeral"}}]
        contents: list[MessageParam] = [MessageParam(role=_ROLE_MAP[message.role], content=message.content) for message in messages]

        kwargs: dict[str, Any] = dict(model=model.model_string, system=system_blocks, messages=contents, **params)
        if output_schema is not None:
            # the schema messages.parse() would send, merged with the effort setting. The answer is checked by
            # LLMClient._answer rather than inside the SDK, which would fail on a refusal before we saw it
            kwargs["output_config"] = {**kwargs.get("output_config", {}), "format": {"type": "json_schema", "schema": transform_schema(output_schema.model_json_schema())}}
        return kwargs

    async def _send(self, kwargs: dict[str, Any]) -> Message:
        # not streamed, so the client timeout caps the whole answer: raise CLAUDE_TIMEOUT for long max-effort runs
        return await self._client.messages.create(**kwargs)

    def _tokens_of(self, response: Message) -> int:
        return _total_tokens(response)

    def _parse(self, model: ClaudeLLMModel, response: Message, output_schema: type[BaseModel] | None) -> LLMResponse:
        output_text = _output_text(response)
        tokens = message_tokens(response.usage)
        reasoning = _thinking_summaries(response)

        if response.stop_reason == "refusal":
            details = response.stop_details
            category = details.category if details is not None else None
            raise self._refused(model, f"category {category}", output_text, reasoning, tokens)

        return self._answer(model, output_text, reasoning, tokens, output_schema)

    def _check_batch_id(self, request_id: str) -> None:
        if not _CUSTOM_ID.match(request_id):
            raise ValueError(f"Claude batch ids are 1-64 letters, digits, '-' or '_'; got {request_id!r}")

    def _batch_size(self, request_id: str, body: dict[str, Any]) -> int:
        # the request's share of the create call's JSON body, which the size limit counts
        return len(json.dumps({"custom_id": request_id, "params": body}, ensure_ascii=False).encode()) + 1

    async def _submit_batch(self, model: ClaudeLLMModel, requests: list[tuple[str, dict[str, Any]]]) -> str:
        # the model is in every request's params (from _request), so the batch itself needs none
        # the params are the kwargs generate() passes to messages.create
        batch = await self._client.messages.batches.create(
            requests=[Request(custom_id=request_id, params=cast(MessageCreateParamsNonStreaming, body)) for request_id, body in requests]
        )
        return batch.id

    async def _batch_ended(self, batch_id: str) -> bool:
        batch = await self._client.messages.batches.retrieve(batch_id)
        return batch.processing_status == "ended"

    async def _batch_results(self, job: BatchJob) -> dict[str, Message | LLMCallError]:
        assert job.batch_id is not None  # BatchRun collects only submitted jobs
        results: dict[str, Message | LLMCallError] = {}
        async for item in await self._client.messages.batches.results(job.batch_id):
            result = item.result
            if result.type == "succeeded":
                results[item.custom_id] = result.message
            elif result.type == "errored":
                error = result.error.error
                results[item.custom_id] = LLMCallError(_BATCH_ERROR_KINDS.get(error.type, FailureKind.PROVIDER), f"{error.type}: {error.message}")
            elif result.type == "expired":
                results[item.custom_id] = LLMCallError(FailureKind.EXPIRED, f"Batch {job.batch_id} expired before this request ran")
            else:
                results[item.custom_id] = LLMCallError(FailureKind.CANCELLED, f"Batch {job.batch_id} was cancelled before this request ran")
        return results

    async def _cancel_batch(self, batch_id: str) -> None:
        await self._client.messages.batches.cancel(batch_id)
