from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from plybench.llm.options import LLMCallOptions, ReasoningEffort
from plybench.llm.rate_limit import ModelLimits, estimate_prompt_tokens
from plybench.llm.tokens import EmbeddingTokens, LLMTokens
from plybench.utils.const import MILLION
from plybench.utils.enums import ExtendedEnum

# output-side token guess used to size a rate reservation when the caller sets no max_tokens; only
# consulted for models that carry a tokens_per_minute quota
DEFAULT_OUTPUT_ESTIMATE = 8_000
# providers cap how many inputs a single embedding request may carry; larger calls are split into batches
DEFAULT_EMBEDDING_BATCH_SIZE = 100
# a retired model stays in its registry so recorded results still resolve and cost; it cannot be called
RETIRED_MESSAGE = "Model {name} is retired: the provider no longer serves it. It stays in the registry only so recorded results still load."


class TemperatureSupport(ExtendedEnum):
    ALWAYS = "always"
    WITHOUT_THINKING = "without_thinking"  # rejected while the model reasons
    NEVER = "never"


class EmbeddingTask(str, ExtendedEnum):
    SEARCH_QUERY = "search_query"
    SEARCH_DOCUMENT = "search_document"
    QUESTION_ANSWERING = "question_answering"
    FACT_CHECKING = "fact_checking"
    CODE_RETRIEVAL = "code_retrieval"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    SEMANTIC_SIMILARITY = "semantic_similarity"


class EmbeddingModel(ABC):
    def __init__(
        self,
        model_name: str,
        model_string: str,
        context_size: int,
        input_cost: float,
        max_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        output_dimensionality: int | None = None,
        truncates_input: bool = False,
        limits: ModelLimits | None = None,
        retired: bool = False,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")

        self.model_name = model_name  # stable internal alias, used in configs/results
        self.model_string = model_string  # exact vendor id sent to the API
        self.context_size = context_size  # max input tokens the model accepts
        self.input_cost = input_cost  # USD per 1M input tokens
        self.max_batch_size = max_batch_size  # max inputs the provider accepts in one request
        # None keeps the provider's native width; a smaller value asks the provider to truncate the vector
        self.output_dimensionality = output_dimensionality
        # providers that truncate oversized inputs themselves opt out of the context guard
        self.truncates_input = truncates_input
        # per-model quota; None means only the provider-wide semaphore applies
        self.limits = limits
        self.retired = retired

    def cost(self, tokens: EmbeddingTokens) -> float:
        return tokens.input_tokens / MILLION * self.input_cost

    def batches(self, texts: list[str]) -> list[list[str]]:
        return [texts[start : start + self.max_batch_size] for start in range(0, len(texts), self.max_batch_size)]

    def check_context(self, texts: list[str]) -> None:
        # a rejected request costs a full retry cycle, so refuse oversized inputs before dispatching
        if self.truncates_input:
            return
        for index, text in enumerate(texts):
            estimate = estimate_prompt_tokens(text)
            if estimate > self.context_size:
                raise ValueError(f"Text at index {index} is ~{estimate} tokens, over the {self.context_size}-token context of {self.model_name}; chunk it before embedding")

    @abstractmethod
    def format_texts(self, texts: list[str], task: EmbeddingTask) -> list[str]:
        raise NotImplementedError


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
        retired: bool = False,
        temperature_support: TemperatureSupport = TemperatureSupport.ALWAYS,
        effort_without_thinking: bool = False,
        needs_effort_to_think: bool = False,
        batch_ratio: float | None = None,
        batch_cached_input_cost: float | None = None,
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
        self.retired = retired
        self.temperature_support = temperature_support
        # the effort also sets answer length, so it means something with thinking off; elsewhere it would be
        # ignored, or would switch reasoning back on
        self.effort_without_thinking = effort_without_thinking
        # thinking on without an effort does not reason: the provider's default effort is no reasoning
        self.needs_effort_to_think = needs_effort_to_think
        # Batch API price as a share of the standard one, for input, cached input and output alike; None when
        # the provider offers no batch price for the model
        self.batch_ratio = batch_ratio
        # a batch cache-read price the provider lists apart from the ratio (some Gemini models)
        self.batch_cached_input_cost = batch_cached_input_cost

    def cost(self, tokens: LLMTokens, batch: bool = False) -> float:
        input_cost, cached_input_cost, output_cost = self.input_cost, self.cached_input_cost, self.output_cost
        if batch:
            if self.batch_ratio is None:
                raise ValueError(f"Model {self.model_name} has no batch price")
            input_cost, output_cost = input_cost * self.batch_ratio, output_cost * self.batch_ratio
            cached_input_cost = self.batch_cached_input_cost if self.batch_cached_input_cost is not None else cached_input_cost * self.batch_ratio

        uncached = max(tokens.input_tokens - tokens.cached_input_tokens, 0) / MILLION * input_cost
        cached = tokens.cached_input_tokens / MILLION * cached_input_cost
        output = tokens.output_tokens / MILLION * output_cost
        return uncached + cached + output

    def _validate_common(self, options: LLMCallOptions) -> None:
        # validation to verify that the options are compatible with the model
        if self.retired:
            raise ValueError(RETIRED_MESSAGE.format(name=self.model_name))

        if options.thinking_enabled and not self.thinking:
            raise ValueError(f"Model {self.model_name} does not support thinking")

        if self.thinking_only and not options.thinking_enabled:
            raise ValueError(f"Model {self.model_name} requires thinking to be enabled")

        if options.reasoning_effort is not None:
            if self.supported_reasoning is None:
                raise ValueError(f"Model {self.model_name} does not accept a reasoning_effort")
            if options.reasoning_effort not in self.supported_reasoning:
                raise ValueError(f"Model {self.model_name} does not support reasoning_effort {options.reasoning_effort!r}; supported: {sorted(self.supported_reasoning)}")
            if not options.thinking_enabled and not self.effort_without_thinking:
                raise ValueError(f"Model {self.model_name} takes a reasoning_effort only with thinking enabled")

        if options.thinking_enabled and options.reasoning_effort is None and self.needs_effort_to_think:
            raise ValueError(f"Model {self.model_name} does not reason without a reasoning_effort; set one with thinking enabled")

        if options.temperature is not None:
            if self.temperature_support is TemperatureSupport.NEVER:
                raise ValueError(f"Model {self.model_name} does not accept temperature")
            if self.temperature_support is TemperatureSupport.WITHOUT_THINKING and options.thinking_enabled:
                raise ValueError(f"Model {self.model_name} accepts temperature only with thinking disabled")

    def validate(self, options: LLMCallOptions) -> None:
        self._validate_common(options)

    @abstractmethod
    def extract_params(self, options: LLMCallOptions) -> Any:
        raise NotImplementedError
