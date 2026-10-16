from __future__ import annotations

import copy
from typing import Any

from plyllm.model import LLMModel
from plyllm.options import LLMCallOptions, ReasoningEffort
from plyllm.utils import ExtendedEnum

# a sampling parameter: temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty
Sampling = dict[str, float | int]


class SchemaDelivery(ExtendedEnum):
    # how an output schema reaches a model on an open-weight server. Under guided decoding the model does not see the
    # schema, so with ENFORCED alone it does not know which fields come next
    ENFORCED = "enforced"  # the server enforces it (response_format)
    PROMPT = "prompt"  # it is in the system prompt only, for a model that loops when it is enforced
    PROMPT_AND_ENFORCED = "prompt_and_enforced"  # in the prompt, and enforced by the server

    @property
    def in_prompt(self) -> bool:
        return self is not SchemaDelivery.ENFORCED

    @property
    def enforced(self) -> bool:
        return self is not SchemaDelivery.PROMPT


class OpenWeightLLMModel(LLMModel):
    # an open-weight model on a server we run (vLLM, Ollama), called through Chat Completions. It has no per-token
    # price. Its sampling is sent with every request: otherwise the server would take it from the checkpoint's
    # generation_config.json (vLLM) or the Modelfile (Ollama), which no result records. Its thinking switches are
    # chat template variables here; a server that switches thinking another way overrides _thinking_fields
    def __init__(
        self,
        model_name: str,
        sampling_thinking: Sampling | None = None,
        sampling_no_thinking: Sampling | None = None,
        template_thinking: dict[str, Any] | None = None,
        template_no_thinking: dict[str, Any] | None = None,
        thinking: bool = False,
        thinking_only: bool = False,
        schema_delivery: SchemaDelivery = SchemaDelivery.ENFORCED,
        supported_reasoning: frozenset[ReasoningEffort] | None = None,
        retired: bool = False,
        needs_effort_to_think: bool = False,
        served_name: str | None = None,
    ) -> None:
        # the served name is the id sent; it defaults to the alias, since vLLM is started under exactly that name
        super().__init__(
            model_name,
            served_name or model_name,
            thinking=thinking,
            thinking_only=thinking_only,
            supported_reasoning=supported_reasoning,
            retired=retired,
            needs_effort_to_think=needs_effort_to_think,
        )
        if thinking and sampling_thinking is None:
            raise ValueError(f"Model {model_name} reasons, so it needs sampling_thinking")
        if not thinking_only and sampling_no_thinking is None:
            raise ValueError(f"Model {model_name} answers without thinking, so it needs sampling_no_thinking")
        self.schema_delivery = schema_delivery
        self.sampling_thinking = sampling_thinking or {}
        self.sampling_no_thinking = sampling_no_thinking or {}
        # chat_template_kwargs for each thinking mode
        self.template_thinking = template_thinking or {}
        self.template_no_thinking = template_no_thinking or {}

    def extract_params(self, options: LLMCallOptions) -> dict[str, Any]:
        self.validate(options)
        thinking = options.thinking_enabled and self.thinking

        # copies, so a caller that edits the request cannot change the registry entry
        body: dict[str, Any] = copy.deepcopy(self.sampling_thinking if thinking else self.sampling_no_thinking)
        if options.temperature is not None:
            body["temperature"] = options.temperature
        body.update(self._thinking_fields(options, thinking))

        # sampling goes in extra_body as a whole: top_k, min_p and repetition_penalty are not SDK arguments, and a
        # temperature given beside an extra_body temperature would be overwritten by it
        params: dict[str, Any] = {"extra_body": body}
        if options.max_tokens:
            params["max_tokens"] = options.max_tokens
        return params

    def _thinking_fields(self, options: LLMCallOptions, thinking: bool) -> dict[str, Any]:
        # the request fields that switch thinking and set its effort
        template = copy.deepcopy(self.template_thinking if thinking else self.template_no_thinking)
        if options.reasoning_effort is not None:
            # the templates read the effort as a template variable, so it goes where they look for it
            template["reasoning_effort"] = options.reasoning_effort
        return {"chat_template_kwargs": template} if template else {}
