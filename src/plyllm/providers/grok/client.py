from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel

from plybench.llm.client import LLMClient
from plybench.llm.llm_config import LLMConfig
from plybench.llm.message import LLMMessage
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.grok.models import GrokLLMModel, grok_models
from plybench.llm.providers.openai.client import _reasoning_summaries, build_response, responses_tokens, responses_total_tokens
from plybench.llm.providers.providers import Provider
from plybench.llm.response import EmbeddingResponse, LLMResponse

_RETRY_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, APIError)


class GrokLLMClient(LLMClient):
    provider_key = Provider.GROK

    def __init__(self, client: AsyncOpenAI, concurrency: int = 10) -> None:
        super().__init__(grok_models(), concurrency)
        self._client = client

    @classmethod
    def build(cls, config: LLMConfig) -> GrokLLMClient | None:
        if config.grok is None:
            return None
        client = AsyncOpenAI(api_key=config.grok.api_key, base_url=config.grok.base_url, timeout=None)
        return cls(client, config.default_concurrency)

    def _should_retry_on_error(self, error: Exception) -> bool:
        return isinstance(error, _RETRY_ERRORS)

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
        output_text = response.output_parsed.model_dump_json() if output_schema is not None else response.output_text
        tokens = responses_tokens(response.usage)

        return build_response(self.provider_key, model.model_string, output_text, reasoning, tokens, output_schema)

    async def embed(self, model_name: str, texts: list[str]) -> EmbeddingResponse:
        raise NotImplementedError("Grok embeddings are not supported in this package")
