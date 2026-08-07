from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from plybench.llm.options import LLMCallOptions, ReasoningEffort
from plybench.llm.rate_limit import ModelLimits
from plybench.llm.tokens import LLMTokens
from plybench.utils.const import MILLION

# output-side token guess used to size a rate reservation when the caller sets no max_tokens; only
# consulted for models that carry a tokens_per_minute quota
DEFAULT_OUTPUT_ESTIMATE = 8_000


class LLMModel(ABC):
    def __init__(
        self,
        model_name: str,
        model_string: str,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        cached_input_cost: float = 0.0,
        thinking: bool = False,
        thinking_only: bool = False,
        can_use_json_schema: bool = True,
        weak_structured_output: bool = False,
        supported_reasoning: frozenset[ReasoningEffort] | None = None,
        limits: ModelLimits | None = None,
        default_output_estimate: int = DEFAULT_OUTPUT_ESTIMATE,
    ) -> None:
        self.model_name = model_name  # stable internal alias, used in configs/results
        self.model_string = model_string  # exact vendor id sent to the API

        self.input_cost = input_cost  # USD per 1M input tokens
        self.output_cost = output_cost  # USD per 1M output tokens
        self.cached_input_cost = cached_input_cost  # USD per 1M input tokens

        self.thinking = thinking  # if the model supports thinking
        self.thinking_only = thinking_only  # if the model only supports thinking
        self.can_use_json_schema = can_use_json_schema  # if the model can use JSON schema for structured output
        # models without robust native structured output need the schema injected into the prompt
        self.weak_structured_output = weak_structured_output
        self.supported_reasoning = supported_reasoning
        # per-model quota; None means only the provider-wide semaphore applies
        self.limits = limits
        self.default_output_estimate = default_output_estimate

    def cost(self, tokens: LLMTokens) -> float:
        uncached_cost = max(tokens.input_tokens - tokens.cached_input_tokens, 0) / MILLION * self.input_cost
        cached_cost = tokens.cached_input_tokens / MILLION * self.cached_input_cost
        output_cost = tokens.output_tokens / MILLION * self.output_cost
        return uncached_cost + cached_cost + output_cost

    def _validate_common(self, options: LLMCallOptions) -> None:
        # validation to verify that the options are compatible with the model
        if options.thinking_enabled and not self.thinking:
            raise ValueError(f"Model {self.model_name} does not support thinking")

        if self.thinking_only and not options.thinking_enabled:
            raise ValueError(f"Model {self.model_name} requires thinking to be enabled")

        if options.reasoning_effort is not None:
            if self.supported_reasoning is None:
                raise ValueError(f"Model {self.model_name} does not accept a reasoning_effort")
            if options.reasoning_effort not in self.supported_reasoning:
                raise ValueError(f"Model {self.model_name} does not support reasoning_effort {options.reasoning_effort!r}; supported: {sorted(self.supported_reasoning)}")

    def validate(self, options: LLMCallOptions) -> None:
        self._validate_common(options)

    @abstractmethod
    def extract_params(self, options: LLMCallOptions) -> Any:
        raise NotImplementedError
