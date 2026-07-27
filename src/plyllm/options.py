from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]


@dataclass(frozen=True)
class LLMCallOptions:
    reasoning_effort: ReasoningEffort | None = None
    thinking_enabled: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
