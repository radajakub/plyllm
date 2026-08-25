from __future__ import annotations

from typing import Any

import httpx
from mistralai.client import Mistral
from mistralai.client.errors import MistralError
from mistralai.client.models import AssistantMessage, ChatCompletionResponse, TextChunk, ThinkChunk, UsageInfo
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel

from plybench.llm.client import LLMClient
from plybench.llm.llm_config import LLMConfig
from plybench.llm.message import LLMMessage
from plybench.llm.model import EmbeddingModel, EmbeddingTask
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.mistral.models import MistralLLMModel, mistral_models
from plybench.llm.providers.providers import Provider
from plybench.llm.response import EmbeddingBatch, EmbeddingResponse, LLMResponse, OutputText, ReasoningTrace
from plybench.llm.tokens import LLMTokens

# MistralError covers every HTTP failure, so the status predicate below decides what is worth
# retrying; NoResponseError and httpx failures are transport-level and always are
_RETRY_ERRORS = (MistralError, httpx.TimeoutException, httpx.TransportError)
_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
# the SDK defaults to 300s per request, which a long thinking trace can exceed
_TIMEOUT_MS = 600_000


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, MistralError):
        return error.status_code in _RETRY_STATUSES
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


class MistralLLMClient(LLMClient):
    provider_key = Provider.MISTRAL

    def __init__(self, client: Mistral, concurrency: int = 10) -> None:
        super().__init__(mistral_models(), [], concurrency)
        self._client = client

    @classmethod
    def build(cls, config: LLMConfig) -> MistralLLMClient | None:
        if config.mistral is None:
            return None
        client = Mistral(api_key=config.mistral.api_key, timeout_ms=_TIMEOUT_MS)
        return cls(client, config.default_concurrency)

    def _should_retry_on_error(self, error: Exception) -> bool:
        return isinstance(error, _RETRY_ERRORS) and _is_retryable(error)

    async def generate(
        self,
        model_name: str,
        system: LLMMessage,
        messages: list[LLMMessage],
        options: LLMCallOptions,
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        model: MistralLLMModel = self.resolve_model(model_name)
        if output_schema is not None and not model.can_use_json_schema:
            raise ValueError(f"Model {model.model_name} does not support JSON schema")

        params = model.extract_params(options)
        # Mistral has no separate system field; the instructions lead the message list
        contents = [system.to_dict(), *(message.to_dict() for message in messages)]

        kwargs: dict[str, Any] = dict(model=model.model_string, messages=contents, prompt_cache_key="PlyBench", **params)
        if output_schema is not None:
            # chat.parse_async() rejects the chunked content reasoning models return, so we send the
            # strict schema it would have built and validate the text ourselves
            kwargs["response_format"] = response_format_from_pydantic_model(output_schema)

        response = await self._dispatch(
            model,
            system,
            messages,
            options,
            lambda: self._client.chat.complete_async(**kwargs),
            _RETRY_ERRORS,
            retry_if=_is_retryable,
            tokens_of=total_tokens,
        )

        choice = response.choices[0] if response.choices else None
        reasoning, output_text = split_content(choice.message if choice is not None else None)
        if output_schema is not None:
            output_text = output_schema.model_validate_json(output_text).model_dump_json()

        items: list[ReasoningTrace | OutputText] = []
        if reasoning:
            items.append(ReasoningTrace(reasoning))
        items.append(OutputText([output_text]))

        return LLMResponse(self.provider_key, model.model_string, completion_tokens(response.usage), items, output_text, output_schema)

    async def embed(self, model_name: str, texts: list[str], task: EmbeddingTask) -> EmbeddingResponse:
        raise NotImplementedError("Mistral embeddings are not supported in this package")

    async def _embed_batch(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        raise NotImplementedError("Mistral embeddings are not supported in this package")
