from __future__ import annotations

from typing import Any

from anthropic import APIConnectionError, APIError, APITimeoutError, AsyncAnthropic, RateLimitError
from anthropic.types import Message, MessageParam, OutputTokensDetails, ParsedMessage, TextBlockParam, Usage
from pydantic import BaseModel

from plybench.llm.client import LLMClient
from plybench.llm.llm_config import LLMConfig
from plybench.llm.message import LLMMessage, MessageRole
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.claude.models import ClaudeLLMModel, claude_models
from plybench.llm.providers.providers import Provider
from plybench.llm.response import EmbeddingResponse, LLMResponse, OutputText, ReasoningTrace
from plybench.llm.tokens import LLMTokens

_RETRY_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, APIError)
# the messages array only carries the conversation; the system prompt is a separate request field
_ROLE_MAP: dict[MessageRole, str] = {"user": "user", "assistant": "assistant", "system": "user"}


def _thinking_summaries(message: Message) -> list[str]:
    return [block.thinking for block in message.content if block.type == "thinking" and block.thinking]


def _output_text(message: Message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def _total_tokens(result: tuple[ParsedMessage[Any], OutputTokensDetails | None]) -> int:
    # what the rate gate charges against a tokens_per_minute quota
    usage = result[0].usage
    return usage.input_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0) + usage.output_tokens


def message_tokens(usage: Usage, output_details: OutputTokensDetails | None) -> LLMTokens:
    # usage.input_tokens counts only the uncached prefix; cache reads and writes are reported separately
    cached_tokens = usage.cache_read_input_tokens or 0
    cache_write_tokens = usage.cache_creation_input_tokens or 0
    return LLMTokens(
        input_tokens=usage.input_tokens + cached_tokens + cache_write_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=output_details.thinking_tokens if output_details is not None else 0,
    )


class ClaudeLLMClient(LLMClient):
    provider_key = Provider.CLAUDE

    def __init__(self, client: AsyncAnthropic, concurrency: int = 10) -> None:
        super().__init__(claude_models(), concurrency)
        self._client = client

    @classmethod
    def build(cls, config: LLMConfig) -> ClaudeLLMClient | None:
        if config.claude is None:
            return None
        client = AsyncAnthropic(api_key=config.claude.api_key, timeout=None)
        return cls(client, config.default_concurrency)

    def _should_retry_on_error(self, error: Exception) -> bool:
        return isinstance(error, _RETRY_ERRORS)

    async def _final_message(self, kwargs: dict[str, Any]) -> tuple[ParsedMessage[Any], OutputTokensDetails | None]:
        # streaming keeps long thinking traces from tripping the request timeout
        async with self._client.messages.stream(**kwargs) as stream:
            output_details: OutputTokensDetails | None = None
            async for event in stream:
                # the accumulated message drops output_tokens_details, so keep the streamed value
                if event.type == "message_delta":
                    output_details = event.usage.output_tokens_details or output_details
            return await stream.get_final_message(), output_details

    async def generate(
        self,
        model_name: str,
        system: LLMMessage,
        messages: list[LLMMessage],
        options: LLMCallOptions,
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        model: ClaudeLLMModel = self.resolve_model(model_name)
        if output_schema is not None and not model.can_use_json_schema:
            raise ValueError(f"Model {model.model_name} does not support JSON schema")

        params = model.extract_params(options)
        # the breakpoint sits on the system prompt, so the varying turns stay outside the cached prefix
        system_blocks: list[TextBlockParam] = [{"type": "text", "text": system.content, "cache_control": {"type": "ephemeral"}}]
        contents: list[MessageParam] = [{"role": _ROLE_MAP[message.role], "content": message.content} for message in messages]

        kwargs: dict[str, Any] = dict(model=model.model_string, system=system_blocks, messages=contents, **params)
        if output_schema is not None:
            kwargs["output_format"] = output_schema

        response, output_details = await self._dispatch(model, system, messages, options, lambda: self._final_message(kwargs), _RETRY_ERRORS, tokens_of=_total_tokens)

        if response.stop_reason == "refusal":
            details = response.stop_details
            category = details.category if details is not None else None
            raise ValueError(f"Model {model.model_string} refused the request (category: {category})")

        parsed_output: BaseModel | None = response.parsed_output if output_schema is not None else None
        output_text = parsed_output.model_dump_json() if parsed_output is not None else _output_text(response)
        tokens = message_tokens(response.usage, output_details)
        reasoning = _thinking_summaries(response)

        items: list[ReasoningTrace | OutputText] = []
        if reasoning:
            items.append(ReasoningTrace(reasoning))
        items.append(OutputText([output_text]))

        return LLMResponse(self.provider_key, model.model_string, tokens, items, output_text, output_schema)

    async def embed(self, model_name: str, texts: list[str]) -> EmbeddingResponse:
        raise NotImplementedError("Claude embeddings are not supported in this package")
