from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMTokens:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    # reasoning_tokens is the reasoning subset of output_tokens (output_tokens is the full completion)
    reasoning_tokens: int = 0

    def __add__(self, other: LLMTokens) -> LLMTokens:
        return LLMTokens(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True)
class EmbeddingTokens:
    input_tokens: int = 0

    def __add__(self, other: EmbeddingTokens) -> EmbeddingTokens:
        return EmbeddingTokens(
            input_tokens=self.input_tokens + other.input_tokens,
        )
