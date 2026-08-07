from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel

from plybench.llm.concurrency import ProviderSemaphore, safe_call
from plybench.llm.llm_config import LLMConfig
from plybench.llm.message import LLMMessage
from plybench.llm.model import LLMModel
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.providers import Provider
from plybench.llm.rate_limit import ModelLimits, RateGate, estimate_prompt_tokens, make_gate
from plybench.llm.response import EmbeddingResponse, LLMResponse
from plybench.llm.tokens import LLMTokens

T = TypeVar("T")


class LLMClient(ABC):
    provider_key: Provider

    def __init__(self, models: list[LLMModel], concurrency: int = 10) -> None:
        self._models: dict[str, LLMModel] = {model.model_name: model for model in models}
        self._semaphore = ProviderSemaphore(concurrency)
        # the provider semaphore is the aggregate ceiling; gates shape each model within it
        self._gates: dict[str, RateGate] = {name: make_gate(model.limits) for name, model in self._models.items()}

    @classmethod
    @abstractmethod
    def build(cls, config: LLMConfig) -> LLMClient | None:
        raise NotImplementedError

    def bootstrap(self) -> None:
        # optional hook: providers that need to download/verify resources (e.g. local models)
        # override this; remote providers keep the no-op default
        return None

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

    def gate(self, model: LLMModel) -> RateGate:
        return self._gates[model.model_name]

    def set_model_limits(self, model_name: str, limits: ModelLimits | None) -> None:
        model = self.resolve_model(model_name)
        model.limits = limits
        # in-flight calls keep the old gate; new ones are shaped by the new limits
        self._gates[model_name] = make_gate(limits)

    def _token_estimate(self, model: LLMModel, system: LLMMessage, messages: list[LLMMessage], options: LLMCallOptions) -> int:
        prompt_tokens = estimate_prompt_tokens(system.content, *(message.content for message in messages))
        return prompt_tokens + (options.max_tokens or model.default_output_estimate)

    async def _dispatch(
        self,
        model: LLMModel,
        system: LLMMessage,
        messages: list[LLMMessage],
        options: LLMCallOptions,
        task: Callable[[], Awaitable[T]],
        retry_errors: tuple[type[Exception], ...],
        retry_if: Callable[[Exception], bool] | None = None,
        tokens_of: Callable[[T], int] | None = None,
    ) -> T:
        estimate = self._token_estimate(model, system, messages, options)
        gate = self.gate(model)
        return await self._semaphore.run(lambda: safe_call(lambda: gate.run(task, estimate, tokens_of), retry_errors=retry_errors, retry_if=retry_if))

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
