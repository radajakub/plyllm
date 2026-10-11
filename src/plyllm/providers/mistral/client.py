from __future__ import annotations

import json
import sys
from typing import Any

import httpx2
from mistralai.client import Mistral
from mistralai.client.errors import MistralError
from mistralai.client.models import AssistantMessage, BaseModelCard, BatchJob, ChatCompletionRequest, ChatCompletionResponse, FTModelCard, TextChunk, ThinkChunk, UsageInfo
from mistralai.client.utils.serializers import marshal_json, unmarshal
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel

from plyllm.batch import BatchJob as StoredBatchJob
from plyllm.client import LLMClient
from plyllm.errors import FailureKind, LLMCallError, retryable_status, status_kind
from plyllm.llm_config import DEFAULT_PROMPT_CACHE_KEY, LLMConfig
from plyllm.message import LLMMessage
from plyllm.options import LLMCallOptions
from plyllm.providers.mistral.models import MistralLLMModel, mistral_models
from plyllm.providers.providers import Provider
from plyllm.response import LLMResponse
from plyllm.tokens import LLMTokens

# MistralError covers every HTTP failure, so the status predicate below decides what is worth
# retrying; NoResponseError and httpx2 failures (the SDK's transport since mistralai 3) are transport-level and always are
_RETRY_ERRORS = (MistralError, httpx2.TimeoutException, httpx2.TransportError)
# the endpoint generate() calls; one batch carries requests to one endpoint
_BATCH_ENDPOINT = "/v1/chat/completions"
# "For batches with less than 10k requests, we support inline batching" (https://docs.mistral.ai/studio/batch-processing);
# larger calls are split into several jobs, of which Mistral allows any number. The docs give no size limit
_MAX_BATCH_REQUESTS = 9_999
_MAX_BATCH_BYTES = sys.maxsize
# a job in any of these has finished with every request it will ever finish
_BATCH_ENDED = frozenset({"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"})
# a job that ended without running every request
_UNRUN_KINDS = {"TIMEOUT_EXCEEDED": FailureKind.EXPIRED, "CANCELLED": FailureKind.CANCELLED, "FAILED": FailureKind.PROVIDER}


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, MistralError):
        return retryable_status(error.status_code)
    return True


def _extra_int(extras: dict[str, Any], group: str, key: str) -> int:
    # cached/reasoning counts are not declared on UsageInfo; they arrive as extra fields
    details = extras.get(group)
    if not isinstance(details, dict):
        return 0
    return details.get(key) or 0


def completion_tokens(usage: UsageInfo) -> LLMTokens:
    extras = usage.additional_properties or {}
    cached = _extra_int(extras, "prompt_tokens_details", "cached_tokens") or (extras.get("num_cached_tokens") or 0)
    return LLMTokens(
        input_tokens=usage.prompt_tokens or 0,
        cached_input_tokens=cached,
        output_tokens=usage.completion_tokens or 0,
        reasoning_tokens=_extra_int(extras, "completion_tokens_details", "reasoning_tokens"),
    )


def total_tokens(response: ChatCompletionResponse) -> int:
    usage = response.usage
    return usage.total_tokens or (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)


def _batch_body(kwargs: dict[str, Any]) -> dict[str, Any]:
    # the JSON chat.complete_async() sends for these kwargs, through the SDK's own request model and serializer
    return json.loads(marshal_json(ChatCompletionRequest(**kwargs), ChatCompletionRequest))


def _batch_line(line: dict[str, Any]) -> ChatCompletionResponse | LLMCallError:
    # one line of a job's output or error file: {"custom_id", "response": {"status_code", "body"}, "error"}. The docs
    # show only the output line, so an error is read as a dict with a message or as plain text
    error = line.get("error")
    if error:
        message = error.get("message", error) if isinstance(error, dict) else error
        return LLMCallError(FailureKind.PROVIDER, str(message))
    response = line.get("response") or {}
    status, body = response.get("status_code"), response.get("body")
    if status != 200 or not isinstance(body, dict):
        message = (body.get("message") or body.get("error")) if isinstance(body, dict) else body
        return LLMCallError(status_kind(status), f"HTTP {status}: {message}")
    # how the SDK reads a chat completion
    return unmarshal(body, ChatCompletionResponse)


def _unrun(remote: BatchJob, job: StoredBatchJob) -> dict[str, LLMCallError]:
    kind = _UNRUN_KINDS.get(remote.status)
    if kind is None:
        return {}
    reasons = "; ".join(f"{error.message} (x{error.count})" for error in remote.errors)
    return {
        request_id: LLMCallError(kind, f"Batch {job.batch_id} ended as {remote.status} before this request ran{': ' + reasons if reasons else ''}") for request_id in job.requests
    }


def split_content(message: AssistantMessage | None) -> tuple[list[str], str]:
    # with reasoning active the content is a chunk list (thinking + text); otherwise a plain string
    if message is None or message.content is None:
        return [], ""
    if isinstance(message.content, str):
        return [], message.content

    reasoning: list[str] = []
    output: list[str] = []
    for chunk in message.content:
        if isinstance(chunk, ThinkChunk):
            reasoning.extend(part.text for part in chunk.thinking if isinstance(part, TextChunk))
        elif isinstance(chunk, TextChunk):
            output.append(chunk.text)
    return reasoning, "".join(output)


class MistralLLMClient(LLMClient[MistralLLMModel, ChatCompletionResponse]):
    provider_key = Provider.MISTRAL
    _retry_errors = _RETRY_ERRORS
    supports_batch = True
    max_batch_requests = _MAX_BATCH_REQUESTS
    max_batch_bytes = _MAX_BATCH_BYTES

    def __init__(self, client: Mistral, concurrency: int | None = None, prompt_cache_key: str = DEFAULT_PROMPT_CACHE_KEY) -> None:
        super().__init__(mistral_models(), [], concurrency)
        self._client = client
        self._prompt_cache_key = prompt_cache_key

    @classmethod
    def build(cls, config: LLMConfig) -> MistralLLMClient | None:
        if config.mistral is None:
            return None
        # the SDK defaults to 300s per request, which a long thinking trace can exceed
        client = Mistral(api_key=config.mistral.api_key, timeout_ms=int(config.mistral.timeout * 1000))
        return cls(client, config.default_concurrency, config.prompt_cache_key)

    def _should_retry_on_error(self, error: Exception) -> bool:
        return isinstance(error, _RETRY_ERRORS) and _is_retryable(error)

    def error_kind(self, error: Exception) -> FailureKind:
        if isinstance(error, httpx2.TimeoutException):
            return FailureKind.TIMEOUT  # before TransportError, which it subclasses
        if isinstance(error, httpx2.TransportError):
            return FailureKind.CONNECTION
        if not isinstance(error, MistralError):
            return FailureKind.OTHER
        return status_kind(error.status_code)

    async def aclose(self) -> None:
        # the SDK closes its pool only on leaving `async with`; this is that exit
        await self._client.__aexit__(None, None, None)

    async def served_models(self) -> set[str]:
        listed = await self._client.models.list_async()
        cards = [model for model in listed.data or [] if isinstance(model, (BaseModelCard, FTModelCard))]
        return {name for card in cards for name in (card.id, *(card.aliases or []))}

    def _request(self, model: MistralLLMModel, system: LLMMessage, messages: list[LLMMessage], options: LLMCallOptions, output_schema: type[BaseModel] | None) -> dict[str, Any]:
        # the request body, apart from the call, so a batch line carries exactly what generate() sends
        params = model.extract_params(options)
        # Mistral has no separate system field; the instructions lead the message list
        contents = [system.to_dict(), *(message.to_dict() for message in messages)]

        kwargs: dict[str, Any] = dict(model=model.model_string, messages=contents, prompt_cache_key=self._prompt_cache_key, **params)
        if output_schema is not None:
            # chat.parse_async() rejects the chunked content reasoning models return, so we send the
            # strict schema it would have built; LLMClient._answer checks the answer
            kwargs["response_format"] = response_format_from_pydantic_model(output_schema)
        return kwargs

    async def _send(self, kwargs: dict[str, Any]) -> ChatCompletionResponse:
        return await self._client.chat.complete_async(**kwargs)

    def _tokens_of(self, response: ChatCompletionResponse) -> int:
        return total_tokens(response)

    def _parse(self, model: MistralLLMModel, response: ChatCompletionResponse, output_schema: type[BaseModel] | None) -> LLMResponse:
        choice = response.choices[0] if response.choices else None
        reasoning, output_text = split_content(choice.message if choice is not None else None)

        return self._answer(model, output_text, reasoning, completion_tokens(response.usage), output_schema)

    def _batch_size(self, request_id: str, body: dict[str, Any]) -> int:
        return len(json.dumps({"custom_id": request_id, "body": _batch_body(body)}, ensure_ascii=False).encode())

    async def _submit_batch(self, model: MistralLLMModel, requests: list[tuple[str, dict[str, Any]]]) -> str:
        # each body keeps its model, as generate() sends it; the docs make it optional once the job names one
        job = await self._client.batch.jobs.create_async(
            endpoint=_BATCH_ENDPOINT,
            model=model.model_string,
            requests=[{"custom_id": request_id, "body": _batch_body(body)} for request_id, body in requests],
        )
        return job.id

    async def _batch_ended(self, batch_id: str) -> bool:
        job = await self._client.batch.jobs.get_async(job_id=batch_id)
        return job.status in _BATCH_ENDED

    async def _batch_results(self, job: StoredBatchJob) -> dict[str, ChatCompletionResponse | LLMCallError]:
        assert job.batch_id is not None  # BatchRun collects only submitted jobs
        remote = await self._client.batch.jobs.get_async(job_id=job.batch_id)
        results: dict[str, ChatCompletionResponse | LLMCallError] = {}
        for file_id in (remote.output_file, remote.error_file):
            if file_id:
                download = await self._client.files.download_async(file_id=file_id)
                try:
                    content = (await download.aread()).decode()
                finally:
                    await download.aclose()
                for text in content.splitlines():
                    if text.strip():
                        line = json.loads(text)
                        results[line["custom_id"]] = _batch_line(line)
        return {**_unrun(remote, job), **results}

    async def _cancel_batch(self, batch_id: str) -> None:
        await self._client.batch.jobs.cancel_async(job_id=batch_id)
