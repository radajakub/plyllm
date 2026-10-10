from __future__ import annotations

from typing import Any

from plybench.llm.model import LLMModel, TemperatureSupport
from plybench.llm.options import LLMCallOptions, ReasoningEffort

# grok-4.5 accepts reasoning_effort of low / medium / high (xhigh is silently treated as high); grok-4.3 adds none,
# which is what thinking off sends
_GROK_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high"})
_BATCH_RATIO = 0.8


class GrokLLMModel(LLMModel):
    def __init__(
        self,
        model_name: str,
        model_string: str,
        input_cost: float,
        output_cost: float,
        cached_input_cost: float = 0.0,
        thinking: bool = False,
        thinking_only: bool = False,
        batch_ratio: float | None = None,
        supported_reasoning: frozenset[ReasoningEffort] | None = None,
        retired: bool = False,
    ) -> None:
        super().__init__(
            model_name,
            model_string,
            input_cost=input_cost,
            output_cost=output_cost,
            cached_input_cost=cached_input_cost,
            thinking=thinking,
            thinking_only=thinking_only,
            supported_reasoning=supported_reasoning,
            retired=retired,
            # the docs do not list temperature for the reasoning models
            temperature_support=TemperatureSupport.NEVER,
            batch_ratio=batch_ratio,
        )

    def extract_params(self, options: LLMCallOptions) -> dict[str, Any]:
        self.validate(options)

        params: dict[str, Any] = {}

        if not options.thinking_enabled:
            # omitting the effort would leave the model reasoning at its default
            params["reasoning"] = {"effort": "none"}
        elif options.reasoning_effort is not None:
            params["reasoning"] = {"summary": "detailed", "effort": options.reasoning_effort}

        if options.max_tokens:
            params["max_output_tokens"] = options.max_tokens

        return params


def grok_models() -> list[GrokLLMModel]:
    # prices are the standard (< 200k context) tier, USD per 1M tokens. The batch discount is per model (20% on
    # grok-4.3 and grok-4.20, on every token type); "models not listed above have no batch discount", and the
    # docs do not say whether grok-4.5 accepts batch requests at all, so it has no batch price
    return [
        # grok-4.5 (reasoning cannot be disabled; effort low/medium/high)
        GrokLLMModel("grok-4.5", "grok-4.5", input_cost=2.0, output_cost=6.0, cached_input_cost=0.3, thinking=True, thinking_only=True, supported_reasoning=_GROK_REASONING),
        # grok-4.3 (default effort low; none turns reasoning off; the model page lists xhigh in one place and not in another)
        GrokLLMModel("grok-4.3", "grok-4.3", input_cost=1.25, output_cost=2.5, cached_input_cost=0.2, thinking=True, batch_ratio=_BATCH_RATIO, supported_reasoning=_GROK_REASONING),
        # grok-4.20 reasoning (always reasons; the docs list no reasoning_effort for it)
        GrokLLMModel(
            "grok-4.20-reasoning",
            "grok-4.20-0309-reasoning",
            input_cost=1.25,
            output_cost=2.5,
            cached_input_cost=0.2,
            thinking=True,
            thinking_only=True,
            batch_ratio=_BATCH_RATIO,
        ),
    ]
