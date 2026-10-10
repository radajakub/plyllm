from __future__ import annotations

from typing import Any, Literal

from anthropic import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncAnthropic, transform_schema
from anthropic.types import Message, MessageParam, TextBlockParam, Usage
from pydantic import BaseModel

from plybench.llm.client import LLMClient
from plybench.llm.errors import FailureKind, retryable_status, status_kind
from plybench.llm.llm_config import LLMConfig
from plybench.llm.message import LLMMessage, MessageRole
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.claude.models import ClaudeLLMModel, claude_models
from plybench.llm.providers.providers import Provider
from plybench.llm.response import LLMResponse
from plybench.llm.tokens import LLMTokens

AnthropicRoles = Literal["user", "assistant", "system"]

# APITimeoutError subclasses APIConnectionError and RateLimitError subclasses APIStatusError, so these two
# cover every failure the SDK raises for a request that was sent
_RETRY_ERRORS = (APIConnectionError, APIStatusError)
# the messages array only carries the conversation; the system prompt is a separate request field
_ROLE_MAP: dict[MessageRole, AnthropicRoles] = {"user": "user", "assistant": "assistant", "system": "user"}


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


class ClaudeLLMClient(LLMClient[ClaudeLLMModel]):
    provider_key = Provider.CLAUDE

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

    async def generate(
        self,
        model_name: str,
        system: LLMMessage,
        messages: list[LLMMessage],
        options: LLMCallOptions,
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        model: ClaudeLLMModel = self._resolve_for_generate(model_name, output_schema)

        params = model.extract_params(options)
        # the breakpoint sits on the system prompt, so the varying turns stay outside the cached prefix
        system_blocks: list[TextBlockParam] = [{"type": "text", "text": system.content, "cache_control": {"type": "ephemeral"}}]
        contents: list[MessageParam] = [MessageParam(role=_ROLE_MAP[message.role], content=message.content) for message in messages]

        kwargs: dict[str, Any] = dict(model=model.model_string, system=system_blocks, messages=contents, **params)
        if output_schema is not None:
            # the schema messages.parse() would send, merged with the effort setting. The answer is checked by
            # LLMClient._answer rather than inside the SDK, which would fail on a refusal before we saw it
            kwargs["output_config"] = {**kwargs.get("output_config", {}), "format": {"type": "json_schema", "schema": transform_schema(output_schema.model_json_schema())}}

        # not streamed, so the client timeout caps the whole answer: raise CLAUDE_TIMEOUT for long max-effort runs
        response = await self._dispatch(model, system, messages, options, lambda: self._client.messages.create(**kwargs), _RETRY_ERRORS, tokens_of=_total_tokens)

        output_text = _output_text(response)
        tokens = message_tokens(response.usage)
        reasoning = _thinking_summaries(response)

        if response.stop_reason == "refusal":
            details = response.stop_details
            category = details.category if details is not None else None
            raise self._refused(model, f"category {category}", output_text, reasoning, tokens)

        return self._answer(model, output_text, reasoning, tokens, output_schema)
