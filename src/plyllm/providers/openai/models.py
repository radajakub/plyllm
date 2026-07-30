from __future__ import annotations

from typing import Any

from plybench.llm.model import LLMModel
from plybench.llm.options import LLMCallOptions, ReasoningEffort

# GPT-5.4 / 5.5 / 5.6 standard models (docs also list none; 5.6 also lists max)
_GPT5_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high", "xhigh"})
# Original GPT-5 mini/nano also accept minimal
_GPT5_LEGACY_REASONING: frozenset[ReasoningEffort] = frozenset({"minimal", "low", "medium", "high", "xhigh"})
# Pro variants: medium / high / xhigh only
_GPT5_PRO_REASONING: frozenset[ReasoningEffort] = frozenset({"medium", "high", "xhigh"})


class OpenAILLMModel(LLMModel):
    def __init__(
        self,
        model_name: str,
        model_string: str,
        input_cost: float,
        output_cost: float,
        cached_input_cost: float = 0.0,
        thinking: bool = False,
        new_api: bool = False,
        supported_reasoning: frozenset[ReasoningEffort] | None = None,
    ) -> None:
        super().__init__(
            model_name,
            model_string,
            input_cost=input_cost,
            output_cost=output_cost,
            cached_input_cost=cached_input_cost,
            thinking=thinking,
            supported_reasoning=supported_reasoning,
        )
        # new_api => uses the Responses API reasoning-style models (no free temperature)
        self.new_api = new_api

    def extract_params(self, options: LLMCallOptions) -> dict[str, Any]:
        self.validate(options)

        params: dict[str, Any] = {}

        if options.reasoning_effort is not None:
            params["reasoning"] = {"summary": "detailed", "effort": options.reasoning_effort}
        elif not (options.thinking_enabled or self.new_api):
            if options.temperature is not None:
                params["temperature"] = options.temperature

        if options.max_tokens:
            params["max_output_tokens"] = options.max_tokens

        return params


def openai_models() -> list[OpenAILLMModel]:
    return [
        # GPT-5.6
        OpenAILLMModel("gpt-5.6-sol", "gpt-5.6-sol", input_cost=5.0, output_cost=30.0, cached_input_cost=0.5, thinking=True, new_api=True, supported_reasoning=_GPT5_REASONING),
        OpenAILLMModel(
            "gpt-5.6-terra", "gpt-5.6-terra", input_cost=2.5, output_cost=15.0, cached_input_cost=0.25, thinking=True, new_api=True, supported_reasoning=_GPT5_REASONING
        ),
        OpenAILLMModel("gpt-5.6-luna", "gpt-5.6-luna", input_cost=1.0, output_cost=6.0, cached_input_cost=0.1, thinking=True, new_api=True, supported_reasoning=_GPT5_REASONING),
        # GPT-5.5
        OpenAILLMModel("gpt-5.5", "gpt-5.5-2026-04-23", input_cost=5.0, output_cost=30.0, cached_input_cost=0.5, thinking=True, new_api=True, supported_reasoning=_GPT5_REASONING),
        OpenAILLMModel("gpt-5.5-pro", "gpt-5.5-pro-2026-04-23", input_cost=30.0, output_cost=180.0, thinking=True, new_api=True, supported_reasoning=_GPT5_PRO_REASONING),
        # GPT-5.4
        OpenAILLMModel("gpt-5.4", "gpt-5.4-2026-03-05", input_cost=2.5, output_cost=15.0, cached_input_cost=0.25, thinking=True, new_api=True, supported_reasoning=_GPT5_REASONING),
        OpenAILLMModel(
            "gpt-5.4-mini", "gpt-5.4-mini-2026-03-17", input_cost=0.75, output_cost=4.5, cached_input_cost=0.075, thinking=True, new_api=True, supported_reasoning=_GPT5_REASONING
        ),
        OpenAILLMModel(
            "gpt-5.4-nano", "gpt-5.4-nano-2026-03-17", input_cost=0.2, output_cost=1.25, cached_input_cost=0.02, thinking=True, new_api=True, supported_reasoning=_GPT5_REASONING
        ),
        OpenAILLMModel("gpt-5.4-pro", "gpt-5.4-pro-2026-03-05", input_cost=30.0, output_cost=180.0, thinking=True, new_api=True, supported_reasoning=_GPT5_PRO_REASONING),
        # GPT-5 (legacy, kept for backward compatibility)
        OpenAILLMModel(
            "gpt-5-mini",
            "gpt-5-mini-2025-08-07",
            input_cost=0.25,
            output_cost=2.0,
            cached_input_cost=0.025,
            thinking=True,
            new_api=True,
            supported_reasoning=_GPT5_LEGACY_REASONING,
        ),
        OpenAILLMModel(
            "gpt-5-nano",
            "gpt-5-nano-2025-08-07",
            input_cost=0.05,
            output_cost=0.4,
            cached_input_cost=0.005,
            thinking=True,
            new_api=True,
            supported_reasoning=_GPT5_LEGACY_REASONING,
        ),
    ]
