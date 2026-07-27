from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from omniplay.llm.concurrency import ProviderSemaphore
from omniplay.llm.llm_config import LLMConfig
from omniplay.llm.message import LLMMessage
from omniplay.llm.model import LLMModel
from omniplay.llm.options import LLMCallOptions
from omniplay.llm.providers.providers import Provider
from omniplay.llm.response import EmbeddingResponse, LLMResponse
from omniplay.llm.tokens import LLMTokens


class LLMClient(ABC):
    provider_key: Provider

    def __init__(self, models: list[LLMModel], concurrency: int = 10) -> None:
        self._models: dict[str, LLMModel] = {model.model_name: model for model in models}
        self._semaphore = ProviderSemaphore(concurrency)

    @classmethod
    @abstractmethod
    def build(cls, config: LLMConfig) -> LLMClient | None:
        raise NotImplementedError

    def resolve_model(self, model_name: str) -> LLMModel:
        model = self._models.get(model_name, None)
        if model is None:
            raise ValueError(f"Model {model_name} not found for provider {self.provider_key.value}")
        return model

    def get_available_models(self) -> list[LLMModel]:
        return list(self._models.values())

    def calculate_cost(self, model_name: str, tokens: LLMTokens) -> float:
        return self.resolve_model(model_name).cost(tokens)

    def set_concurrency(self, concurrency: int | None) -> None:
        self._semaphore.configure(concurrency)

    @abstractmethod
    def _should_retry_on_error(self, error: Exception) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        model_name: str,
        system: LLMMessage,
        messages: list[LLMMessage],
        options: LLMCallOptions,
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def embed(self, model_name: str, texts: list[str]) -> EmbeddingResponse:
        raise NotImplementedError
