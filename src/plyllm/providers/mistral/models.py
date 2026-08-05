from __future__ import annotations

from typing import Any

from plybench.llm.model import LLMModel
from plybench.llm.options import LLMCallOptions, ReasoningEffort

# the SDK's ReasoningEffort enum minus "none" (which is what we send when thinking is off, so it is
# not an effort a caller picks) and without OpenAI's "max" tier
_MISTRAL_REASONING: frozenset[ReasoningEffort] = frozenset({"minimal", "low", "medium", "high", "xhigh"})

# reasoning_effort is mandatory on these models; "none" suppresses the thinking chunk entirely
_THINKING_OFF = "none"
_DEFAULT_EFFORT: ReasoningEffort = "high"


class MistralLLMModel(LLMModel):
    def extract_params(self, options: LLMCallOptions) -> dict[str, Any]:
        self.validate(options)

        params: dict[str, Any] = {}

        if self.thinking:
            params["reasoning_effort"] = (options.reasoning_effort or _DEFAULT_EFFORT) if options.thinking_enabled else _THINKING_OFF

        if options.temperature is not None:
            params["temperature"] = options.temperature

        if options.max_tokens:
            params["max_tokens"] = options.max_tokens

        # prompt_mode="reasoning" would prepend Mistral's own reasoning system prompt, which would
        # compete with the benchmark's; leaving it unset keeps our instructions authoritative
        return params


def mistral_models() -> list[MistralLLMModel]:
    # no rate limits are declared here: Mistral's per-model TPM/RPS quotas are account-specific, so
    # they are the caller's to apply via LLM.set_model_limits() (see scripts/_shared.py)
    return [
        MistralLLMModel(
            "mistral-medium-3.5",
            "mistral-medium-2604",
            input_cost=1.5,
            output_cost=7.5,
            cached_input_cost=0.15,
            thinking=True,
            supported_reasoning=_MISTRAL_REASONING,
        ),
        MistralLLMModel(
            "mistral-small-4",
            "mistral-small-2603",
            input_cost=0.15,
            output_cost=0.6,
            cached_input_cost=0.015,
            thinking=True,
            supported_reasoning=_MISTRAL_REASONING,
        ),
    ]
