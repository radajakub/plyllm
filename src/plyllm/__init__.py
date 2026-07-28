from omniplay.llm.llm_config import (
    GeminiProviderConfig,
    LLMConfig,
    MetacentrumProviderConfig,
    OpenAIProviderConfig,
)
from omniplay.llm.message import LLMMessage, MessageRole
from omniplay.llm.model import LLMModel
from omniplay.llm.model_config import ModelConfig
from omniplay.llm.options import LLMCallOptions, ReasoningEffort
from omniplay.llm.providers.providers import Provider
from omniplay.llm.response import (
    EmbeddingResponse,
    LLMResponse,
    LLMResponseItem,
    OutputText,
    ReasoningTrace,
)
from omniplay.llm.tokens import LLMTokens

__all__ = [
    "LLM",
    "LLMConfig",
    "OpenAIProviderConfig",
    "GeminiProviderConfig",
    "MetacentrumProviderConfig",
    "LLMMessage",
    "MessageRole",
    "LLMModel",
    "ModelConfig",
    "LLMCallOptions",
    "Provider",
    "ReasoningEffort",
    "LLMResponse",
    "LLMResponseItem",
    "ReasoningTrace",
    "OutputText",
    "EmbeddingResponse",
    "LLMTokens",
]


# The LLM router pulls the provider SDKs (openai / google-genai), so expose it lazily: importing the
# light LLM value types (LLMMessage, ModelConfig, LLMResponse, ...) -- e.g. from core.prompt_adapter --
# must not drag in those SDKs. `from omniplay.llm import LLM` still works (triggers this on access).
def __getattr__(name: str) -> object:
    if name == "LLM":
        from omniplay.llm.router import LLM

        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
