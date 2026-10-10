from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI

# private: the helper responses.parse() uses to build the strict schema; a test pins it against SDK upgrades
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import BaseModel

from plybench.llm.client import LLMClient
from plybench.llm.errors import FailureKind, retryable_status, status_kind
from plybench.llm.llm_config import LLMConfig
from plybench.llm.message import LLMMessage
from plybench.llm.model import EmbeddingModel
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.openai.models import OpenAILLMModel, openai_embedding_models, openai_models
from plybench.llm.providers.providers import Provider
from plybench.llm.response import EmbeddingBatch, LLMResponse
from plybench.llm.tokens import EmbeddingTokens, LLMTokens

# APITimeoutError subclasses APIConnectionError and RateLimitError subclasses APIStatusError, so these two
# cover every failure the SDK raises for a request that was sent
_RETRY_ERRORS = (APIConnectionError, APIStatusError)


def _reasoning_summaries(response: Any) -> list[str]:
    return [summary.text for item in response.output if item.type == "reasoning" for summary in item.summary]


def _refusal(response: Any) -> str | None:
    # a declined request comes back as a refusal content item instead of output text
    refusals = [content.refusal for item in response.output if item.type == "message" for content in item.content if content.type == "refusal"]
    return "\n".join(refusals) if refusals else None


def responses_total_tokens(response: Any) -> int:
    # what the rate gate charges against a tokens_per_minute quota
    usage = response.usage
    if usage is None:
        return 0
    return (usage.input_tokens or 0) + (usage.output_tokens or 0)


def responses_tokens(usage: Any) -> LLMTokens:
    if usage is None:
        return LLMTokens(input_tokens=0, output_tokens=0, cached_input_tokens=0, reasoning_tokens=0)
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return LLMTokens(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cached_input_tokens=getattr(input_details, "cached_tokens", 0) or 0,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) or 0,
    )


class OpenAILLMClient(LLMClient[OpenAILLMModel]):
    provider_key = Provider.OPENAI

    def __init__(self, client: AsyncOpenAI, concurrency: int | None = None) -> None:
        super().__init__(openai_models(), openai_embedding_models(), concurrency)
        self._client = client

    @classmethod
    def build(cls, config: LLMConfig) -> OpenAILLMClient | None:
        if config.openai is None:
            return None
        client = AsyncOpenAI(
            api_key=config.openai.api_key,
            organization=config.openai.organization,
            project=config.openai.project,
            timeout=config.openai.timeout,
        )
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
        model = self._resolve_for_generate(model_name, output_schema)

        params = model.extract_params(options)
        kwargs: dict[str, Any] = dict(
            model=model.model_string,
            instructions=system.content,
            input=[message.to_dict() for message in messages],
            store=False,
            prompt_cache_key="PlyBench",
            **params,
        )
        if output_schema is not None:
            # the strict schema responses.parse() would send; the answer is checked by LLMClient._answer
            # rather than inside the SDK, so one that does not fit keeps its raw text and tokens
            kwargs["text"] = {"format": type_to_text_format_param(output_schema)}

        response = await self._dispatch(model, system, messages, options, lambda: self._client.responses.create(**kwargs), _RETRY_ERRORS, tokens_of=responses_total_tokens)

        reasoning = _reasoning_summaries(response)
        tokens = responses_tokens(response.usage)

        refusal = _refusal(response)
        if refusal is not None:
            raise self._refused(model, refusal, refusal, reasoning, tokens)

        return self._answer(model, response.output_text, reasoning, tokens, output_schema)

    async def _embed_batch(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        response = await self._dispatch_embedding(
            model,
            texts,
            lambda: (
                self._client.embeddings.create(model=model.model_string, input=texts, dimensions=model.output_dimensionality)
                if model.output_dimensionality is not None
                else self._client.embeddings.create(model=model.model_string, input=texts)
            ),
            _RETRY_ERRORS,
            tokens_of=lambda result: result.usage.total_tokens if result.usage is not None else 0,
        )

        # the API may return items out of order, so sort by the index it echoes back
        items = sorted(response.data, key=lambda item: item.index)
        total_tokens = response.usage.total_tokens if response.usage is not None else 0
        return EmbeddingBatch([item.embedding for item in items], EmbeddingTokens(total_tokens))
