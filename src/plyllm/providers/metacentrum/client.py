from __future__ import annotations

import json
import re
import warnings
from typing import Any

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel

from plybench.llm.client import DEFAULT_RETRIES, LLMClient
from plybench.llm.errors import FailureKind, retryable_status, status_kind
from plybench.llm.llm_config import LLMConfig
from plybench.llm.message import LLMMessage
from plybench.llm.model import EmbeddingModel, EmbeddingTask
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.metacentrum.models import MetacentrumLLMModel, metacentrum_models
from plybench.llm.providers.providers import Provider
from plybench.llm.response import EmbeddingBatch, EmbeddingResponse, LLMResponse
from plybench.llm.tokens import LLMTokens

# APITimeoutError subclasses APIConnectionError and RateLimitError subclasses APIStatusError, so these two
# cover every failure the SDK raises for a request that was sent
_RETRY_ERRORS = (APIConnectionError, APIStatusError)


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
            if content.type in ["output_text", "reasoning_text"] and content.text is not None:
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


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _schema_instructions(system: str, output_schema: type[BaseModel]) -> str:
    schema = json.dumps(output_schema.model_json_schema(), indent=1)
    return f"{system}\n\nReply with a single JSON object matching this schema, and nothing else -- no prose, no code fence:\n{schema}"


def _json_body(text: str) -> str:
    return _FENCE.sub("", text).strip()


def silence_proxy_serializer_warnings() -> None:
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)


class MetacentrumLLMClient(LLMClient[MetacentrumLLMModel]):
    provider_key = Provider.METACENTRUM

    def __init__(self, client: AsyncOpenAI, concurrency: int = 4, retries: int = DEFAULT_RETRIES) -> None:
        super().__init__(metacentrum_models(), [], concurrency, retries)
        self._client = client

    @classmethod
    def build(cls, config: LLMConfig) -> MetacentrumLLMClient | None:
        if config.metacentrum is None:
            return None
        silence_proxy_serializer_warnings()
        client = AsyncOpenAI(
            api_key=config.metacentrum.api_key,
            base_url=config.metacentrum.base_url,
            timeout=config.metacentrum.timeout,
            max_retries=config.metacentrum.max_retries,
        )
        return cls(client, config.default_concurrency, config.metacentrum.retries)

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
        model: MetacentrumLLMModel = self.resolve_model(model_name)
        if output_schema is not None and not model.can_use_json_schema:
            raise ValueError(f"Model {model.model_name} does not support JSON schema")

        params = model.extract_params(options)
        extra_body = model.extract_extra_body(options)
        # a schema this model cannot have enforced without looping is asked for in the prompt instead
        ask_in_prompt = output_schema is not None and model.weak_structured_output
        instructions = _schema_instructions(system.content, output_schema) if ask_in_prompt and output_schema else system.content
        kwargs: dict[str, Any] = dict(
            model=model.model_string,
            instructions=instructions,
            input=[message.to_dict() for message in messages],
            store=False,
            extra_body=extra_body,
            **params,
        )
        if output_schema is not None and not ask_in_prompt:
            kwargs["text_format"] = output_schema

        enforced = output_schema is not None and not ask_in_prompt
        method = self._client.responses.parse if enforced else self._client.responses.create

        response = await self._dispatch(model, system, messages, options, lambda: method(**kwargs), _RETRY_ERRORS, tokens_of=responses_total_tokens)

        if enforced:
            reasoning = [content.text for item in response.output if item.type == "reasoning" for content in item.content or [] if content.text is not None]
            output_text = response.output_parsed.model_dump_json() if response.output_parsed is not None else response.output_text
        else:
            # also the path for an asked-for schema, so the inline <think> block a hosted model may emit
            # is stripped before anything tries to read the answer as JSON
            output_text, reasoning = _extract_text_and_reasoning(response)
            if ask_in_prompt:
                output_text = _json_body(output_text)

        tokens = responses_tokens(response.usage)
        return LLMResponse.from_parts(self.provider_key, model.model_string, output_text, reasoning, tokens, output_schema)

    async def embed(self, model_name: str, texts: list[str], task: EmbeddingTask) -> EmbeddingResponse:
        raise NotImplementedError("Metacentrum embeddings are not supported in this package")

    async def _embed_batch(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        raise NotImplementedError("Metacentrum embeddings are not supported in this package")
