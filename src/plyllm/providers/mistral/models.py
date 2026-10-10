from __future__ import annotations

from typing import Any

from plybench.llm.model import LLMModel
from plybench.llm.options import LLMCallOptions, ReasoningEffort

# the docs document only "high" and "none" (sent when thinking is off), and mistral-small-2603 answers
# 400 to "low": "Must be one of (none, high)". The SDK enum lists more levels than the models accept
_MISTRAL_REASONING: frozenset[ReasoningEffort] = frozenset({"high"})

# reasoning_effort is mandatory on these models; "none" suppresses the thinking chunk entirely
_THINKING_OFF = "none"
_DEFAULT_EFFORT: ReasoningEffort = "high"
# "Batch processing ... reduces the price by 50%"; the docs do not say whether that covers cached input too
_BATCH_RATIO = 0.5


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
        # sale price with no announced end date; the list price is 1.36 / 4.18 (cached 0.14)
        MistralLLMModel(
            "mistral-large-4",
            "mistral-large-4",
            input_cost=0.68,
            output_cost=2.09,
            cached_input_cost=0.07,
            thinking=True,
            supported_reasoning=_MISTRAL_REASONING,
            batch_ratio=_BATCH_RATIO,
        ),
        MistralLLMModel(
            "mistral-medium-3.5",
            "mistral-medium-2604",
            input_cost=1.5,
            output_cost=7.5,
            cached_input_cost=0.15,
            thinking=True,
            supported_reasoning=_MISTRAL_REASONING,
            batch_ratio=_BATCH_RATIO,
        ),
        MistralLLMModel(
            "mistral-small-4",
            "mistral-small-2603",
            input_cost=0.15,
            output_cost=0.6,
            cached_input_cost=0.015,
            thinking=True,
            supported_reasoning=_MISTRAL_REASONING,
            batch_ratio=_BATCH_RATIO,
        ),
    ]
