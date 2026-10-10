from __future__ import annotations

from typing import Any

from plybench.llm.model import EmbeddingModel, EmbeddingTask, LLMModel, TemperatureSupport
from plybench.llm.options import LLMCallOptions, ReasoningEffort

# GPT-5.4 / 5.5 standard models (docs also list none, which is what thinking off sends)
_GPT5_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high", "xhigh"})
# GPT-5.6 standard models additionally accept max
_GPT56_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high", "xhigh", "max"})
# Original GPT-5 mini/nano also accept minimal
_GPT5_LEGACY_REASONING: frozenset[ReasoningEffort] = frozenset({"minimal", "low", "medium", "high", "xhigh"})
# Pro variants: medium / high / xhigh only
_GPT5_PRO_REASONING: frozenset[ReasoningEffort] = frozenset({"medium", "high", "xhigh"})
_GPT6_REASONING: frozenset[ReasoningEffort] = frozenset({"low", "medium", "high", "xhigh", "max"})
# the embeddings endpoint accepts up to 2048 inputs per request
_EMBEDDING_BATCH_SIZE = 2048


class OpenAILLMModel(LLMModel):
    def __init__(
        self,
        model_name: str,
        model_string: str,
        input_cost: float,
        output_cost: float,
        cached_input_cost: float = 0.0,
        thinking: bool = False,
        thinking_only: bool = False,
        needs_effort_to_think: bool = False,
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
            # temperature is accepted only at effort none, which is how thinking is turned off
            temperature_support=TemperatureSupport.NEVER if thinking_only else TemperatureSupport.WITHOUT_THINKING,
            needs_effort_to_think=needs_effort_to_think,
            # every model's Batch row is half its Standard row, cached input included
            batch_ratio=0.5,
        )

    def extract_params(self, options: LLMCallOptions) -> dict[str, Any]:
        self.validate(options)

        params: dict[str, Any] = {}

        if options.thinking_enabled:
            # without the summary the reasoning stays hidden, whatever the effort
            params["reasoning"] = {"summary": "detailed"}
            if options.reasoning_effort is not None:
                params["reasoning"]["effort"] = options.reasoning_effort
        else:
            # omitting the effort would leave the model reasoning at its default
            params["reasoning"] = {"effort": "none"}
            if options.temperature is not None:
                params["temperature"] = options.temperature

        if options.max_tokens:
            params["max_output_tokens"] = options.max_tokens

        return params


def openai_models() -> list[OpenAILLMModel]:
    # thinking_only: the model page lists no effort none, so thinking cannot be turned off.
    # needs_effort_to_think: the default effort is none, so thinking on needs an explicit effort
    return [
        # GPT-6
        OpenAILLMModel(
            "gpt-6-astra", "gpt-6-astra", input_cost=10.0, output_cost=50.0, cached_input_cost=1.0, thinking=True, thinking_only=True, supported_reasoning=_GPT6_REASONING
        ),
        OpenAILLMModel("gpt-6-sol", "gpt-6-sol", input_cost=2.0, output_cost=10.0, cached_input_cost=0.2, thinking=True, supported_reasoning=_GPT6_REASONING),
        OpenAILLMModel("gpt-6-luna", "gpt-6-luna", input_cost=0.1, output_cost=0.5, cached_input_cost=0.01, thinking=True, supported_reasoning=_GPT6_REASONING),
        # GPT-6.1
        OpenAILLMModel(
            "gpt-6.1-sol", "gpt-6.1-sol", input_cost=2.0, output_cost=10.0, cached_input_cost=0.1, thinking=True, thinking_only=True, supported_reasoning=_GPT6_REASONING
        ),
        # GPT-5.6
        OpenAILLMModel("gpt-5.6-sol", "gpt-5.6-sol", input_cost=4.0, output_cost=20.0, cached_input_cost=0.4, thinking=True, supported_reasoning=_GPT56_REASONING),
        OpenAILLMModel("gpt-5.6-terra", "gpt-5.6-terra", input_cost=2.0, output_cost=12.0, cached_input_cost=0.2, thinking=True, supported_reasoning=_GPT56_REASONING),
        OpenAILLMModel("gpt-5.6-luna", "gpt-5.6-luna", input_cost=0.2, output_cost=1.2, cached_input_cost=0.02, thinking=True, supported_reasoning=_GPT56_REASONING),
        # GPT-5.5
        OpenAILLMModel("gpt-5.5", "gpt-5.5-2026-04-23", input_cost=5.0, output_cost=30.0, cached_input_cost=0.5, thinking=True, supported_reasoning=_GPT5_REASONING),
        OpenAILLMModel("gpt-5.5-pro", "gpt-5.5-pro-2026-04-23", input_cost=30.0, output_cost=180.0, thinking=True, thinking_only=True, supported_reasoning=_GPT5_PRO_REASONING),
        # GPT-5.4
        OpenAILLMModel(
            "gpt-5.4",
            "gpt-5.4-2026-03-05",
            input_cost=2.5,
            output_cost=15.0,
            cached_input_cost=0.25,
            thinking=True,
            needs_effort_to_think=True,
            supported_reasoning=_GPT5_REASONING,
        ),
        OpenAILLMModel(
            "gpt-5.4-mini",
            "gpt-5.4-mini-2026-03-17",
            input_cost=0.75,
            output_cost=4.5,
            cached_input_cost=0.075,
            thinking=True,
            needs_effort_to_think=True,
            supported_reasoning=_GPT5_REASONING,
        ),
        # deprecated by OpenAI; shuts down 2027-04-01 (replacement: gpt-6-luna)
        OpenAILLMModel(
            "gpt-5.4-nano",
            "gpt-5.4-nano-2026-03-17",
            input_cost=0.2,
            output_cost=1.25,
            cached_input_cost=0.02,
            thinking=True,
            needs_effort_to_think=True,
            supported_reasoning=_GPT5_REASONING,
        ),
        OpenAILLMModel("gpt-5.4-pro", "gpt-5.4-pro-2026-03-05", input_cost=30.0, output_cost=180.0, thinking=True, thinking_only=True, supported_reasoning=_GPT5_PRO_REASONING),
        # GPT-5 (legacy, kept for backward compatibility; OpenAI shuts both down on 2026-12-11). Their pages no
        # longer list efforts; they reason at their default (medium) when no effort is sent, so thinking_only
        OpenAILLMModel(
            "gpt-5-mini",
            "gpt-5-mini-2025-08-07",
            input_cost=0.25,
            output_cost=2.0,
            cached_input_cost=0.025,
            thinking=True,
            thinking_only=True,
            supported_reasoning=_GPT5_LEGACY_REASONING,
        ),
        OpenAILLMModel(
            "gpt-5-nano",
            "gpt-5-nano-2025-08-07",
            input_cost=0.05,
            output_cost=0.4,
            cached_input_cost=0.005,
            thinking=True,
            thinking_only=True,
            supported_reasoning=_GPT5_LEGACY_REASONING,
        ),
    ]


class OpenAIEmbeddingModel(EmbeddingModel):
    def format_texts(self, texts: list[str], task: EmbeddingTask) -> list[str]:
        # OpenAI embeddings carry no task conditioning; the text goes out as-is
        return texts


def openai_embedding_models() -> list[EmbeddingModel]:
    return [
        OpenAIEmbeddingModel("text-embedding-3-small", "text-embedding-3-small", context_size=8191, input_cost=0.02, max_batch_size=_EMBEDDING_BATCH_SIZE),
        OpenAIEmbeddingModel("text-embedding-3-large", "text-embedding-3-large", context_size=8191, input_cost=0.13, max_batch_size=_EMBEDDING_BATCH_SIZE),
    ]
