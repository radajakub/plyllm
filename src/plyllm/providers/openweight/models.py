from __future__ import annotations

import copy
from typing import Any

from plyllm.model import LLMModel
from plyllm.options import LLMCallOptions, ReasoningEffort

# a sampling parameter: temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty
Sampling = dict[str, float | int]


class OpenWeightLLMModel(LLMModel):
    # an open-weight model on a server we run (vLLM now, Ollama later), called through Chat Completions. It has no
    # per-token price. Its switches are chat template variables, and its sampling is sent with every request:
    # otherwise vLLM would take it from the checkpoint's generation_config.json, which no result records
    def __init__(
        self,
        served_name: str,
        sampling_thinking: Sampling | None = None,
        sampling_no_thinking: Sampling | None = None,
        template_thinking: dict[str, Any] | None = None,
        template_no_thinking: dict[str, Any] | None = None,
        thinking: bool = False,
        thinking_only: bool = False,
        weak_structured_output: bool = False,
        supported_reasoning: frozenset[ReasoningEffort] | None = None,
        retired: bool = False,
    ) -> None:
        # the served name is both the alias and the id sent: the server is started under exactly this name
        super().__init__(
            served_name,
            served_name,
            thinking=thinking,
            thinking_only=thinking_only,
            weak_structured_output=weak_structured_output,
            supported_reasoning=supported_reasoning,
            retired=retired,
        )
        if thinking and sampling_thinking is None:
            raise ValueError(f"Model {served_name} reasons, so it needs sampling_thinking")
        if not thinking_only and sampling_no_thinking is None:
            raise ValueError(f"Model {served_name} answers without thinking, so it needs sampling_no_thinking")
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
        template = copy.deepcopy(self.template_thinking if thinking else self.template_no_thinking)
        if options.reasoning_effort is not None:
            # the templates read the effort as a template variable, so it goes where they look for it
            template["reasoning_effort"] = options.reasoning_effort
        if template:
            body["chat_template_kwargs"] = template

        # sampling goes in extra_body as a whole: top_k, min_p and repetition_penalty are not SDK arguments, and a
        # temperature given beside an extra_body temperature would be overwritten by it
        params: dict[str, Any] = {"extra_body": body}
        if options.max_tokens:
            params["max_tokens"] = options.max_tokens
        return params
