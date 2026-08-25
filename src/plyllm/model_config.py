from dataclasses import dataclass, field

from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.providers import Provider


@dataclass(frozen=True)
class ModelConfig:
    provider: Provider
    model_name: str
    options: LLMCallOptions = field(default_factory=LLMCallOptions)


@dataclass(frozen=True)
class EmbeddingModelConfig:
    provider: Provider
    model_name: str
