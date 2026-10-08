from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel

from plybench.llm.client import LLMClient
from plybench.llm.errors import FailureKind, retryable_status, status_kind
from plybench.llm.llm_config import LLMConfig
from plybench.llm.message import LLMMessage
from plybench.llm.model import EmbeddingModel, EmbeddingTask
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.grok.models import GrokLLMModel, grok_models
from plybench.llm.providers.providers import Provider
from plybench.llm.response import EmbeddingBatch, EmbeddingResponse, LLMResponse
from plybench.llm.tokens import LLMTokens

# APITimeoutError subclasses APIConnectionError and RateLimitError subclasses APIStatusError, so these two
# cover every failure the SDK raises for a request that was sent
_RETRY_ERRORS = (APIConnectionError, APIStatusError)


def _reasoning_summaries(response: Any) -> list[str]:
    return [summary.text for item in response.output if item.type == "reasoning" for summary in item.summary]


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


class GrokLLMClient(LLMClient[GrokLLMModel]):
    provider_key = Provider.GROK

    def __init__(self, client: AsyncOpenAI, concurrency: int = 10) -> None:
        super().__init__(grok_models(), [], concurrency)
        self._client = client

    @classmethod
    def build(cls, config: LLMConfig) -> GrokLLMClient | None:
        if config.grok is None:
            return None
        client = AsyncOpenAI(api_key=config.grok.api_key, base_url=config.grok.base_url, timeout=config.grok.timeout)
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

    async def generate(
        self,
        model_name: str,
        system: LLMMessage,
        messages: list[LLMMessage],
        options: LLMCallOptions,
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        model: GrokLLMModel = self.resolve_model(model_name)
        if output_schema is not None and not model.can_use_json_schema:
            raise ValueError(f"Model {model.model_name} does not support JSON schema")

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
            kwargs["text_format"] = output_schema

        method = self._client.responses.parse if output_schema is not None else self._client.responses.create

        response = await self._dispatch(model, system, messages, options, lambda: method(**kwargs), _RETRY_ERRORS, tokens_of=responses_total_tokens)

        reasoning = _reasoning_summaries(response)
        output_text = response.output_parsed.model_dump_json() if output_schema is not None and response.output_parsed is not None else response.output_text
        tokens = responses_tokens(response.usage)

        return LLMResponse.from_parts(self.provider_key, model.model_string, output_text, reasoning, tokens, output_schema)

    async def embed(self, model_name: str, texts: list[str], task: EmbeddingTask) -> EmbeddingResponse:
        raise NotImplementedError("Grok embeddings are not supported in this package")

    async def _embed_batch(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        raise NotImplementedError("Grok embeddings are not supported in this package")
