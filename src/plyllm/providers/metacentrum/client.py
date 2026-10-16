from __future__ import annotations

import warnings
from typing import Any

from openai import AsyncOpenAI

# private: the helper responses.parse() uses to build the strict schema; a test pins it against SDK upgrades
from openai.lib._parsing._responses import type_to_text_format_param
from openai.types.responses import Response
from pydantic import BaseModel

from plyllm.client import DEFAULT_RETRIES
from plyllm.llm_config import LLMConfig
from plyllm.message import LLMMessage
from plyllm.options import LLMCallOptions
from plyllm.providers.metacentrum.models import MetacentrumLLMModel, metacentrum_models
from plyllm.providers.openweight.common import OpenWeightClientBase, asks_in_prompt, enforces, json_body, schema_instructions, split_think
from plyllm.providers.providers import Provider
from plyllm.response import LLMResponse
from plyllm.tokens import LLMTokens


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

    # some hosted models emit an inline <think>...</think> block instead of reasoning items
    text, inline = split_think("".join(text_parts))
    if inline:
        reasoning_summaries.append(inline)

    return text, reasoning_summaries


def silence_proxy_serializer_warnings() -> None:
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)


class MetacentrumLLMClient(OpenWeightClientBase[MetacentrumLLMModel, Response]):
    provider_key = Provider.METACENTRUM
    # the shared e-INFRA endpoint answered 429 to 10 parallel calls on 2026-10-09
    default_concurrency = 4

    def __init__(self, client: AsyncOpenAI, concurrency: int | None = None, retries: int = DEFAULT_RETRIES) -> None:
        super().__init__(metacentrum_models(), client, concurrency, retries)

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

    def _request(
        self, model: MetacentrumLLMModel, system: LLMMessage, messages: list[LLMMessage], options: LLMCallOptions, output_schema: type[BaseModel] | None
    ) -> dict[str, Any]:
        params = model.extract_params(options)
        extra_body = model.extract_extra_body(options)
        instructions = schema_instructions(system.content, output_schema) if asks_in_prompt(model.schema_delivery, output_schema) and output_schema else system.content
        kwargs: dict[str, Any] = dict(
            model=model.model_string,
            instructions=instructions,
            input=[message.to_dict() for message in messages],
            store=False,
            extra_body=extra_body,
            **params,
        )
        if enforces(model.schema_delivery, output_schema) and output_schema is not None:
            # the strict schema responses.parse() would send; the answer is checked by LLMClient._answer
            # rather than inside the SDK, so one that does not fit keeps its raw text and tokens
            kwargs["text"] = {"format": type_to_text_format_param(output_schema)}
        return kwargs

    async def _send(self, kwargs: dict[str, Any]) -> Response:
        return await self._client.responses.create(**kwargs)

    def _tokens_of(self, response: Response) -> int:
        return responses_total_tokens(response)

    def _parse(self, model: MetacentrumLLMModel, response: Response, output_schema: type[BaseModel] | None) -> LLMResponse:
        # every path strips the inline <think> block a hosted model may emit before the answer is read as JSON
        output_text, reasoning = _extract_text_and_reasoning(response)
        if asks_in_prompt(model.schema_delivery, output_schema):
            output_text = json_body(output_text)
        tokens = responses_tokens(response.usage)

        refusal = _refusal(response)
        if refusal is not None:
            raise self._refused(model, refusal, refusal, reasoning, tokens)

        return self._answer(model, output_text, reasoning, tokens, output_schema)
