from __future__ import annotations

from typing import Any

from plybench.llm.model import LLMModel
from plybench.llm.options import LLMCallOptions, ReasoningEffort

# grok-4.5 accepts reasoning_effort of low / medium / high
_GROK_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high"})


class GrokLLMModel(LLMModel):
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
        # new_api => reasoning-style model (reasons by default, no free temperature)
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


def grok_models() -> list[GrokLLMModel]:
    # prices are the standard (< 200k context) tier, USD per 1M tokens
    return [
        # grok-4.5 (reasoning cannot be disabled; effort low/medium/high)
        GrokLLMModel("grok-4.5", "grok-4.5", input_cost=2.0, output_cost=6.0, cached_input_cost=0.3, thinking=True, new_api=True, supported_reasoning=_GROK_REASONING),
        # grok-4.3 (non-reasoning)
        GrokLLMModel("grok-4.3", "grok-4.3", input_cost=1.25, output_cost=2.5, cached_input_cost=0.2, thinking=True, new_api=True, supported_reasoning=_GROK_REASONING),
        # grok-4.20 reasoning
        GrokLLMModel(
            "grok-4.20-reasoning",
            "grok-4.20-0309-reasoning",
            input_cost=1.25,
            output_cost=2.5,
            cached_input_cost=0.2,
            thinking=True,
            new_api=True,
            supported_reasoning=_GROK_REASONING,
        ),
    ]
