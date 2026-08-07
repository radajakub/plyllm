from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel

from plybench.llm.client import LLMClient
from plybench.llm.llm_config import LLMConfig
from plybench.llm.message import LLMMessage
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.metacentrum.models import MetacentrumLLMModel, metacentrum_models
from plybench.llm.providers.openai.client import build_response, responses_tokens, responses_total_tokens
from plybench.llm.providers.providers import Provider
from plybench.llm.response import EmbeddingResponse, LLMResponse

_RETRY_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, APIError)


def _extract_text_and_reasoning(response: Any) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    reasoning_summaries: list[str] = []
    for output in response.output:
        if output.type == "message":
            target = text_parts
        elif output.type == "reasoning":
            target = reasoning_summaries
        else:
            continue
        for content in output.content:
            if content.type == "output_text" and content.text is not None:
                target.append(content.text)

    text = "".join(text_parts)
    # some hosted models emit an inline <think>...</think> block instead of reasoning items
    if "</think>" in text:
        reasoning, tail = text.rsplit("</think>", 1)
        reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()
        text = tail.strip()
        if reasoning:
            reasoning_summaries.append(reasoning)

    return text, reasoning_summaries


class MetacentrumLLMClient(LLMClient):
    provider_key = Provider.METACENTRUM

    def __init__(self, client: AsyncOpenAI, concurrency: int = 4) -> None:
        super().__init__(metacentrum_models(), concurrency)
        self._client = client

    @classmethod
    def build(cls, config: LLMConfig) -> MetacentrumLLMClient | None:
        if config.metacentrum is None:
            return None
        client = AsyncOpenAI(api_key=config.metacentrum.api_key, base_url=config.metacentrum.base_url)
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
        model: MetacentrumLLMModel = self.resolve_model(model_name)
        if output_schema is not None and not model.can_use_json_schema:
            raise ValueError(f"Model {model.model_name} does not support JSON schema")

        params = model.extract_params(options)
        extra_body = model.extract_extra_body(options)
        kwargs: dict[str, Any] = dict(
            model=model.model_string,
            instructions=system.content,
            input=[message.to_dict() for message in messages],
            store=False,
            extra_body=extra_body,
            **params,
        )
        if output_schema is not None:
            kwargs["text_format"] = output_schema

        method = self._client.responses.parse if output_schema is not None else self._client.responses.create

        response = await self._dispatch(model, system, messages, options, lambda: method(**kwargs), _RETRY_ERRORS, tokens_of=responses_total_tokens)

        if output_schema is not None:
            reasoning = [content.text for item in response.output if item.type == "reasoning" for content in item.content if content.text is not None]
            output_text = response.output_parsed.model_dump_json()
        else:
            output_text, reasoning = _extract_text_and_reasoning(response)

        tokens = responses_tokens(response.usage)
        return build_response(self.provider_key, model.model_string, output_text, reasoning, tokens, output_schema)

    async def embed(self, model_name: str, texts: list[str]) -> EmbeddingResponse:
        raise NotImplementedError("Metacentrum embeddings are not supported in this package")
