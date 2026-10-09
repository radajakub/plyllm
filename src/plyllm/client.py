from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from plybench.llm.concurrency import ProviderSemaphore, safe_call
from plybench.llm.errors import FailureKind, LLMCallError, as_call_error
from plybench.llm.llm_config import LLMConfig
from plybench.llm.message import LLMMessage
from plybench.llm.model import EmbeddingModel, EmbeddingTask, LLMModel
from plybench.llm.options import LLMCallOptions
from plybench.llm.providers.providers import Provider
from plybench.llm.rate_limit import ModelLimits, RateGate, estimate_prompt_tokens, make_gate
from plybench.llm.response import EmbeddingBatch, EmbeddingResponse, LLMResponse
from plybench.llm.tokens import EmbeddingTokens, LLMTokens

# `safe_call` keeps this as its own default; naming it here is what lets a client lower it
DEFAULT_RETRIES = 10

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=LLMModel)


class LLMClient(ABC, Generic[ModelT]):
    provider_key: Provider

    def __init__(self, models: Sequence[ModelT], embedding_models: Sequence[EmbeddingModel], concurrency: int = 10, retries: int = DEFAULT_RETRIES) -> None:
        self._models: dict[str, ModelT] = {model.model_name: model for model in models}
        self._embedding_models: dict[str, EmbeddingModel] = {model.model_name: model for model in embedding_models}
        self._semaphore = ProviderSemaphore(concurrency)
        # the SDKs retry inside each of these, so the two counts multiply: a provider that stalls needs a
        # lower number here rather than a longer timeout there
        self._retries = retries
        # the provider semaphore is the aggregate ceiling; gates shape each model within it
        self._gates: dict[str, RateGate] = {name: make_gate(model.limits) for name, model in self._models.items()}
        # embedding endpoints carry their own quotas, so they get gates independent of the chat models
        self._embedding_gates: dict[str, RateGate] = {name: make_gate(model.limits) for name, model in self._embedding_models.items()}

    @classmethod
    @abstractmethod
    def build(cls, config: LLMConfig) -> LLMClient | None:
        raise NotImplementedError

    def bootstrap(self) -> None:
        # optional hook: providers that need to download/verify resources (e.g. local models)
        # override this; remote providers keep the no-op default
        return None

    def resolve_model(self, model_name: str) -> ModelT:
        model = self._models.get(model_name, None)
        if model is None:
            raise ValueError(f"Model {model_name} not found for provider {self.provider_key.value}")
        return model

    def resolve_embedding_model(self, model_name: str) -> EmbeddingModel:
        model = self._embedding_models.get(model_name, None)
        if model is None:
            raise ValueError(f"Embedding model {model_name} not found for provider {self.provider_key.value}")
        return model

    def get_available_models(self) -> list[ModelT]:
        return list(self._models.values())

    def get_available_embedding_models(self) -> list[EmbeddingModel]:
        return list(self._embedding_models.values())

    def calculate_cost(self, model_name: str, tokens: LLMTokens) -> float:
        return self.resolve_model(model_name).cost(tokens)

    def calculate_embedding_cost(self, model_name: str, tokens: EmbeddingTokens) -> float:
        return self.resolve_embedding_model(model_name).cost(tokens)

    def set_concurrency(self, concurrency: int | None) -> None:
        self._semaphore.configure(concurrency)

    def gate(self, model: LLMModel) -> RateGate:
        return self._gates[model.model_name]

    def embedding_gate(self, model: EmbeddingModel) -> RateGate:
        return self._embedding_gates[model.model_name]

    def set_model_limits(self, model_name: str, limits: ModelLimits | None) -> None:
        model = self.resolve_model(model_name)
        model.limits = limits
        # in-flight calls keep the old gate; new ones are shaped by the new limits
        self._gates[model_name] = make_gate(limits)

    def set_embedding_model_limits(self, model_name: str, limits: ModelLimits | None) -> None:
        model = self.resolve_embedding_model(model_name)
        model.limits = limits
        self._embedding_gates[model_name] = make_gate(limits)

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
        tokens_of: Callable[[T], int] | None = None,
    ) -> T:
        estimate = self._token_estimate(model, system, messages, options)
        gate = self.gate(model)
        try:
            # generation and embeddings share one retry rule per provider: `_should_retry_on_error`
            return await self._semaphore.run(
                lambda: safe_call(lambda: gate.run(task, estimate, tokens_of), retry_errors=retry_errors, retry_if=self._should_retry_on_error, retries=self._retries)
            )
        except Exception as error:
            # classified here, where the SDK's own types are in scope. Callers get a kind without
            # importing six SDKs, and `__cause__` still carries the original for anything that wants it
            raise as_call_error(error, self.error_kind(error)) from error

    async def _dispatch_embedding(
        self,
        model: EmbeddingModel,
        texts: list[str],
        task: Callable[[], Awaitable[T]],
        retry_errors: tuple[type[Exception], ...],
        tokens_of: Callable[[T], int] | None = None,
    ) -> T:
        # embeddings have no output side, so the reservation is the input estimate alone
        estimate = estimate_prompt_tokens(*texts)
        gate = self.embedding_gate(model)
        try:
            return await self._semaphore.run(
                lambda: safe_call(lambda: gate.run(task, estimate, tokens_of), retry_errors=retry_errors, retry_if=self._should_retry_on_error, retries=self._retries)
            )
        except Exception as error:
            raise as_call_error(error, self.error_kind(error)) from error

    def _answer(self, model: LLMModel, output_text: str, reasoning: list[str], tokens: LLMTokens, output_schema: type[BaseModel] | None) -> LLMResponse:
        if output_schema is None:
            return LLMResponse.from_parts(self.provider_key, model.model_string, output_text, reasoning, tokens, None)

        try:
            normalised = output_schema.model_validate_json(output_text).model_dump_json()
        except ValidationError as error:
            raw = LLMResponse.from_parts(self.provider_key, model.model_string, output_text, reasoning, tokens, None)
            raise LLMCallError(FailureKind.UNPARSEABLE, f"{type(error).__name__}: {error}", raw) from error
        return LLMResponse.from_parts(self.provider_key, model.model_string, normalised, reasoning, tokens, output_schema)

    def _refused(self, model: LLMModel, reason: str, output_text: str, reasoning: list[str], tokens: LLMTokens) -> LLMCallError:
        answer = LLMResponse.from_parts(self.provider_key, model.model_string, output_text, reasoning, tokens, None)
        return LLMCallError(FailureKind.REFUSAL, f"Model {model.model_string} refused the request: {reason}", answer)

    @abstractmethod
    def _should_retry_on_error(self, error: Exception) -> bool:
        raise NotImplementedError

    @abstractmethod
    def error_kind(self, error: Exception) -> FailureKind:
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

    async def embed(self, model_name: str, texts: list[str], task: EmbeddingTask) -> EmbeddingResponse:
        # task formatting, context guarding and batching are provider-independent; _embed_batch only
        # has to turn one ready-to-send batch into vectors
        model = self.resolve_embedding_model(model_name)
        if not texts:
            return EmbeddingResponse(self.provider_key, model.model_string, [], EmbeddingTokens())

        formatted = model.format_texts(texts, task)
        if len(formatted) != len(texts):
            raise ValueError(f"format_texts for {model.model_name} returned {len(formatted)} texts for {len(texts)} inputs")
        model.check_context(formatted)

        batches = await asyncio.gather(*(self._embed_batch(model, batch) for batch in model.batches(formatted)))

        embeddings = [vector for batch in batches for vector in batch.embeddings]
        if len(embeddings) != len(texts):
            # without this the caller silently pairs vectors with the wrong texts
            raise ValueError(f"{self.provider_key.value} returned {len(embeddings)} embeddings for {len(texts)} inputs; vectors cannot be aligned to their inputs")

        tokens = sum((batch.tokens for batch in batches), EmbeddingTokens())
        return EmbeddingResponse(self.provider_key, model.model_string, embeddings, tokens)

    @abstractmethod
    async def _embed_batch(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        raise NotImplementedError
