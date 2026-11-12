from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, overload

from pydantic import BaseModel

from plyllm.decision import DecisionAnswer
from plyllm.providers.providers import Provider
from plyllm.tokens import EmbeddingTokens, LLMTokens

T = TypeVar("T", bound=BaseModel)

# the finish reasons that mean the output limit cut the answer: Claude (also a full context window), Gemini, Chat Completions (OpenAI-compatible, Mistral) and the Responses API
_LENGTH_REASONS = frozenset({"max_tokens", "model_context_window_exceeded", "MAX_TOKENS", "length", "model_length", "max_output_tokens"})


@dataclass(frozen=True)
class ReasoningTrace:
    summaries: list[str]


@dataclass(frozen=True)
class OutputText:
    contents: list[str]


LLMResponseItem = ReasoningTrace | OutputText


@dataclass(frozen=True)
class LLMResponse:
    provider: Provider
    # the exact vendor model id that produced this response (for experiment reproducibility)
    model_string: str
    tokens: LLMTokens
    # ordered content items (reasoning summaries + output text) as returned by the provider
    items: list[LLMResponseItem]
    # flattened final text; holds the JSON string when a structured output schema was used
    output_text: str
    # the pydantic schema the output_text conforms to, when structured output was requested
    structured_output_type: type[BaseModel] | None = None
    # answered through a provider's Batch API, so priced at its batch rate: calculate_cost(..., batch=response.batch)
    batch: bool = False
    # the sampling seed the request carried; None when none was asked for or the provider's API takes none
    seed: int | None = None
    # why the provider stopped, as its API names it (e.g. "end_turn", "max_tokens", "STOP", "length"); None when it reports none
    finish_reason: str | None = None
    # the transport retries this call took before its answer arrived (0 for a batch request)
    retries: int = 0

    @classmethod
    def from_parts(
        cls,
        provider: Provider,
        model_string: str,
        output_text: str,
        reasoning: list[str],
        tokens: LLMTokens,
        output_schema: type[BaseModel] | None,
    ) -> LLMResponse:
        # reasoning first, then the answer: the order the providers return them in
        items: list[LLMResponseItem] = [ReasoningTrace(reasoning)] if reasoning else []
        items.append(OutputText([output_text]))
        return cls(provider, model_string, tokens, items, output_text, output_schema)

    @property
    def truncated(self) -> bool:
        # the answer was cut at the output limit, by any provider's name for it
        return self.finish_reason in _LENGTH_REASONS

    @property
    def reasoning(self) -> list[str]:
        return [summary for item in self.items if isinstance(item, ReasoningTrace) for summary in item.summaries]

    @overload
    def resolve_structured_output(self, model: type[T]) -> T: ...

    @overload
    def resolve_structured_output(self, model: None = None) -> BaseModel | None: ...

    def resolve_structured_output(self, model: type[BaseModel] | None = None) -> BaseModel | None:
        schema = model if model is not None else self.structured_output_type
        if schema is None:
            return None
        return schema.model_validate_json(self.output_text)


@dataclass(frozen=True)
class EmbeddingBatch:
    # one provider request worth of vectors, merged by LLMClient.embed into an EmbeddingResponse
    embeddings: list[list[float]]
    tokens: EmbeddingTokens


@dataclass(frozen=True)
class EmbeddingResponse:
    provider: Provider
    model_string: str
    embeddings: list[list[float]]
    tokens: EmbeddingTokens


@dataclass(frozen=True)
class DecisionResponse:
    provider: Provider
    model_string: str
    # one per question, in question order
    answers: list[DecisionAnswer]
    # all of them as the provider reports them, though a decision may bill only input_tokens (see DecisionModel.cost)
    tokens: LLMTokens

    def answer(self, name: str) -> DecisionAnswer:
        for answer in self.answers:
            if answer.name == name:
                return answer
        raise KeyError(f"No answer for question {name}")
