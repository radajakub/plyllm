from __future__ import annotations

from typing import Any

from anthropic.types import ThinkingConfigParam

from plybench.llm.model import LLMModel
from plybench.llm.options import LLMCallOptions, ReasoningEffort

# Claude Opus 5 / Opus 4.8 / Sonnet 5 / Fable 5 accept the whole effort ladder
_EFFORT_FULL: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high", "xhigh", "max"})
# Sonnet 4.6 predates the xhigh level
_EFFORT_NO_XHIGH: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high", "max"})
# Haiku 4.5 rejects output_config.effort; the level is translated into a thinking budget instead
_LEGACY_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high"})

_BUDGET_BY_EFFORT: dict[ReasoningEffort, int] = {
    "low": 4096,
    "medium": 8192,
    "high": 16384,
}
# the API requires 1024 <= budget_tokens < max_tokens
_MIN_THINKING_BUDGET = 1024

# thinking can only be turned off at effort high or below; xhigh / max reject it
_THINKING_ONLY_EFFORTS: frozenset[ReasoningEffort] = frozenset({"xhigh", "max"})


class ClaudeLLMModel(LLMModel):
    def __init__(
        self,
        model_name: str,
        model_string: str,
        input_cost: float,
        output_cost: float,
        cached_input_cost: float = 0.0,
        thinking: bool = False,
        thinking_only: bool = False,
        uses_effort: bool = True,
        supports_temperature: bool = False,
        max_output_tokens: int = 32000,
        supported_reasoning: frozenset[ReasoningEffort] | None = None,
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
        )
        # uses_effort => reasoning depth is set with output_config.effort and thinking is adaptive;
        # legacy models take a numeric thinking budget instead and reject effort
        self.uses_effort = uses_effort
        # the effort-based models reject temperature / top_p / top_k outright
        self.supports_temperature = supports_temperature
        # max_tokens is mandatory on every Anthropic request and caps thinking + answer together
        self.max_output_tokens = max_output_tokens

    def validate(self, options: LLMCallOptions) -> None:
        self._validate_common(options)

        if not options.thinking_enabled and options.reasoning_effort in _THINKING_ONLY_EFFORTS:
            raise ValueError(f"Model {self.model_name} cannot disable thinking at reasoning_effort {options.reasoning_effort!r}")

    def _thinking_config(self, options: LLMCallOptions, max_tokens: int) -> ThinkingConfigParam:
        if not options.thinking_enabled:
            return {"type": "disabled"}

        if self.uses_effort:
            # display defaults to "omitted" on these models, which returns empty thinking blocks
            return {"type": "adaptive", "display": "summarized"}

        effort: ReasoningEffort = options.reasoning_effort or "high"
        budget = min(_BUDGET_BY_EFFORT[effort], max_tokens - _MIN_THINKING_BUDGET)
        return {"type": "enabled", "budget_tokens": max(budget, _MIN_THINKING_BUDGET)}

    def extract_params(self, options: LLMCallOptions) -> dict[str, Any]:
        self.validate(options)

        max_tokens = options.max_tokens or self.max_output_tokens

        params: dict[str, Any] = {
            "max_tokens": max_tokens,
            "thinking": self._thinking_config(options, max_tokens),
        }

        if self.uses_effort and options.reasoning_effort is not None:
            params["output_config"] = {"effort": options.reasoning_effort}

        # sampling is rejected by the effort-based models, and by extended thinking on the legacy ones
        if self.supports_temperature and not options.thinking_enabled and options.temperature is not None:
            params["temperature"] = options.temperature

        return params


def claude_models() -> list[ClaudeLLMModel]:
    # prices are USD per 1M tokens; cached_input_cost is the cache-read rate (0.1x input)
    return [
        # Claude Fable 5 (thinking cannot be disabled; requires 30-day data retention on the org)
        ClaudeLLMModel(
            "claude-fable-5", "claude-fable-5", input_cost=10.0, output_cost=50.0, cached_input_cost=1.0, thinking=True, thinking_only=True, supported_reasoning=_EFFORT_FULL
        ),
        # Claude Opus 5 (thinking is adaptive by default)
        ClaudeLLMModel("claude-opus-5", "claude-opus-5", input_cost=5.0, output_cost=25.0, cached_input_cost=0.5, thinking=True, supported_reasoning=_EFFORT_FULL),
        ClaudeLLMModel("claude-opus-4.8", "claude-opus-4-8", input_cost=5.0, output_cost=25.0, cached_input_cost=0.5, thinking=True, supported_reasoning=_EFFORT_FULL),
        # Claude Sonnet 5
        ClaudeLLMModel("claude-sonnet-5", "claude-sonnet-5", input_cost=3.0, output_cost=15.0, cached_input_cost=0.3, thinking=True, supported_reasoning=_EFFORT_FULL),
        ClaudeLLMModel(
            "claude-sonnet-4.6",
            "claude-sonnet-4-6",
            input_cost=3.0,
            output_cost=15.0,
            cached_input_cost=0.3,
            thinking=True,
            supports_temperature=True,
            supported_reasoning=_EFFORT_NO_XHIGH,
        ),
        # Claude Haiku 4.5 (no effort parameter, numeric thinking budget, 64k output cap)
        ClaudeLLMModel(
            "claude-haiku-4.5",
            "claude-haiku-4-5",
            input_cost=1.0,
            output_cost=5.0,
            cached_input_cost=0.1,
            thinking=True,
            uses_effort=False,
            supports_temperature=True,
            max_output_tokens=16000,
            supported_reasoning=_LEGACY_REASONING,
        ),
    ]
