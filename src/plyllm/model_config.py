from dataclasses import dataclass, field

from omniplay.llm.options import LLMCallOptions
from omniplay.llm.providers.providers import Provider


@dataclass(frozen=True)
class ModelConfig:
    provider: Provider
    model_name: str
    options: LLMCallOptions = field(default_factory=LLMCallOptions)
