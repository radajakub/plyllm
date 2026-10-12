from __future__ import annotations

from typing import Any

# private: the helper chat.completions.parse() uses to build the strict schema; a test pins it against SDK upgrades
from openai.lib._parsing._completions import type_to_response_format_param
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from plyllm.message import LLMMessage
from plyllm.options import LLMCallOptions
from plyllm.providers.openweight.common import OpenWeightClientBase, asks_in_prompt, json_body, schema_instructions, split_think
from plyllm.providers.openweight.models import OpenWeightLLMModel
from plyllm.response import LLMResponse
from plyllm.tokens import LLMTokens


def _text_and_reasoning(message: ChatCompletionMessage) -> tuple[str, list[str]]:
    # a server started with a reasoning parser returns the reasoning in its own field. vLLM renamed it from
    # reasoning_content to reasoning, and a server may send both with the same text, so the first one set is read.
    # The SDK keeps fields it does not model as attributes
    parsed = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    reasoning: list[str] = [parsed] if parsed else []
    # without a parser the <think> block stays inline in the content
    text, inline = split_think(message.content or "")
    if inline:
        reasoning.append(inline)
    return text, reasoning


def _tokens(usage: CompletionUsage | None) -> LLMTokens:
    if usage is None:
        return LLMTokens()
    # vLLM fills the details only when the server is started with --enable-prompt-tokens-details
    cached = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else None
    reasoning = usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else None
    return LLMTokens(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, cached_input_tokens=cached or 0, reasoning_tokens=reasoning or 0)


class OpenWeightLLMClient(OpenWeightClientBase[OpenWeightLLMModel, ChatCompletion]):
    # Chat Completions on a server we run, with the sampling and template switches of our registry. A subclass sets
    # the provider, its registry and build()

    def _request(self, model: OpenWeightLLMModel, system: LLMMessage, messages: list[LLMMessage], options: LLMCallOptions, output_schema: type[BaseModel] | None) -> dict[str, Any]:
        params = model.extract_params(options)
        ask_in_prompt = asks_in_prompt(model, output_schema)
        instructions = schema_instructions(system.content, output_schema) if ask_in_prompt and output_schema else system.content
        kwargs: dict[str, Any] = dict(
            model=model.model_string,
            messages=[LLMMessage.system(instructions).to_dict(), *(message.to_dict() for message in messages)],
            **params,
        )
        if output_schema is not None and not ask_in_prompt:
            # the strict schema chat.completions.parse() would send; the answer is checked by LLMClient._answer
            # rather than inside the SDK, so one that does not fit keeps its raw text and tokens
            kwargs["response_format"] = type_to_response_format_param(output_schema)
        return kwargs

    async def _send(self, kwargs: dict[str, Any]) -> ChatCompletion:
        return await self._client.chat.completions.create(**kwargs)

    def _tokens_of(self, response: ChatCompletion) -> int:
        return response.usage.total_tokens if response.usage else 0

    def _parse(self, model: OpenWeightLLMModel, response: ChatCompletion, output_schema: type[BaseModel] | None) -> LLMResponse:
        message = response.choices[0].message
        output_text, reasoning = _text_and_reasoning(message)
        if asks_in_prompt(model, output_schema):
            output_text = json_body(output_text)
        tokens = _tokens(response.usage)

        if message.refusal:
            raise self._refused(model, message.refusal, output_text, reasoning, tokens)

        return self._answer(model, output_text, reasoning, tokens, output_schema)
