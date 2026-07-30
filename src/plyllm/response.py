from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

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

    @property
    def reasoning(self) -> list[str]:
        return [summary for item in self.items if isinstance(item, ReasoningTrace) for summary in item.summaries]

    def resolve_structured_output(self, model: type[T] | None = None) -> T | None:
        schema = model if model is not None else self.structured_output_type
        if schema is None:
            return None
        return schema.model_validate_json(self.output_text)


@dataclass(frozen=True)
class EmbeddingResponse:
    provider: Provider
    model_string: str
    embeddings: list[list[float]]
    tokens: EmbeddingTokens
