from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    role: MessageRole
    content: str

    @staticmethod
    def system(content: str) -> LLMMessage:
        return LLMMessage("system", content)

    @staticmethod
    def user(content: str) -> LLMMessage:
        return LLMMessage("user", content)

    @staticmethod
    def assistant(content: str) -> LLMMessage:
        return LLMMessage("assistant", content)

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}
