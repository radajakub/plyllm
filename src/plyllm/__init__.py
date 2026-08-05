from plybench.llm.llm_config import (
    DEFAULT_CONCURRENCY,
    ClaudeProviderConfig,
    GeminiProviderConfig,
    GrokProviderConfig,
    HuggingFaceProviderConfig,
    LLMConfig,
    MetacentrumProviderConfig,
    MistralProviderConfig,
    OpenAIProviderConfig,
)
from plybench.llm.message import LLMMessage, MessageRole
from plybench.llm.model import LLMModel
from plybench.llm.model_config import ModelConfig
from plybench.llm.options import LLMCallOptions, ReasoningEffort
from plybench.llm.providers.providers import Provider
from plybench.llm.rate_limit import ModelLimits
from plybench.llm.response import (
    EmbeddingResponse,
    LLMResponse,
    LLMResponseItem,
    OutputText,
    ReasoningTrace,
)
from plybench.llm.tokens import LLMTokens

__all__ = [
    "LLM",
    "LLMConfig",
    "DEFAULT_CONCURRENCY",
    "OpenAIProviderConfig",
    "GeminiProviderConfig",
    "GrokProviderConfig",
    "ClaudeProviderConfig",
    "MistralProviderConfig",
    "MetacentrumProviderConfig",
    "HuggingFaceProviderConfig",
    "LLMMessage",
    "MessageRole",
    "LLMModel",
    "ModelConfig",
    "ModelLimits",
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
# must not drag in those SDKs. `from plybench.llm import LLM` still works (triggers this on access).
def __getattr__(name: str) -> object:
    if name == "LLM":
        from plybench.llm.router import LLM

        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
