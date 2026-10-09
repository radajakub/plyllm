from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, overload

from pydantic import BaseModel

from plybench.llm.providers.providers import Provider
from plybench.llm.tokens import EmbeddingTokens, LLMTokens

T = TypeVar("T", bound=BaseModel)


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
