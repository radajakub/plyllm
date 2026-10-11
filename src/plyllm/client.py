from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from plyllm.batch import BatchJob
from plyllm.concurrency import ProviderSemaphore, safe_call
from plyllm.errors import FailureKind, LLMCallError, as_call_error
from plyllm.llm_config import DEFAULT_CONCURRENCY, LLMConfig
from plyllm.message import LLMMessage
from plyllm.model import RETIRED_MESSAGE, EmbeddingModel, EmbeddingTask, LLMModel
from plyllm.options import LLMCallOptions
from plyllm.providers.providers import Provider
from plyllm.rate_limit import ModelLimits, RateGate, estimate_prompt_tokens, make_gate
from plyllm.response import EmbeddingBatch, EmbeddingResponse, LLMResponse
from plyllm.tokens import EmbeddingTokens, LLMTokens

# `safe_call` keeps this as its own default; naming it here is what lets a client lower it
DEFAULT_RETRIES = 10

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=LLMModel)
# the SDK object a provider's generation call returns, which _parse reads
ResponseT = TypeVar("ResponseT")


class LLMClient(ABC, Generic[ModelT, ResponseT]):
    provider_key: Provider
    # in-flight requests when the caller sets no concurrency; a provider with a tighter quota lowers it
    default_concurrency: int = DEFAULT_CONCURRENCY
    # the SDK exceptions a failed generation may be retried on; `_should_retry_on_error` decides which instances are
    _retry_errors: tuple[type[Exception], ...]
    # a provider with a batch API sets these and implements the _batch hooks; BatchRun refuses the others
    supports_batch: bool = False
    # what one provider batch may hold: requests, and bytes as _batch_size counts them. A larger call is split
    # into several batches
    max_batch_requests: int
    max_batch_bytes: int

    def __init__(self, models: Sequence[ModelT], embedding_models: Sequence[EmbeddingModel], concurrency: int | None = None, retries: int = DEFAULT_RETRIES) -> None:
        self._models: dict[str, ModelT] = {model.model_name: model for model in models}
        self._embedding_models: dict[str, EmbeddingModel] = {model.model_name: model for model in embedding_models}
        self._semaphore = ProviderSemaphore(concurrency if concurrency is not None else self.default_concurrency)
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

    async def aclose(self) -> None:
        # releases the SDK's connection pool. Call it on the event loop that made the requests: a pool left to
        # the garbage collector after asyncio.run() returns logs "Event loop is closed"
        return None

    async def served_models(self) -> set[str]:
        # the model ids the provider's endpoint lists right now, to catch a retired or mistyped model_string
        # before it fails a run. Remote providers override this with their SDK's list call, which is free
        raise NotImplementedError(f"{self.provider_key.value} cannot list the models it serves")

    def resolve_model(self, model_name: str) -> ModelT:
        model = self._models.get(model_name, None)
        if model is None:
            raise ValueError(f"Model {model_name} not found for provider {self.provider_key.value}")
        return model

    def _resolve_for_generate(self, model_name: str, output_schema: type[BaseModel] | None) -> ModelT:
        model = self.resolve_model(model_name)
        if output_schema is not None and not model.can_use_json_schema:
            raise ValueError(f"Model {model.model_name} does not support JSON schema")
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

    def calculate_cost(self, model_name: str, tokens: LLMTokens, batch: bool = False) -> float:
        return self.resolve_model(model_name).cost(tokens, batch)

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

    async def generate(
        self,
        model_name: str,
        system: LLMMessage,
        messages: list[LLMMessage],
        options: LLMCallOptions,
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        # the same sequence for every provider. The steps are kept apart so a batch request carries exactly the
        # body generate() sends and its results are read the same way: the same body means the same measurement
        model = self._resolve_for_generate(model_name, output_schema)
        kwargs = self._request(model, system, messages, options, output_schema)
        response = await self._dispatch(model, system, messages, options, lambda: self._send(kwargs), self._retry_errors, tokens_of=self._tokens_of)
        return self._parse(model, response, output_schema)

    @abstractmethod
    def _request(self, model: ModelT, system: LLMMessage, messages: list[LLMMessage], options: LLMCallOptions, output_schema: type[BaseModel] | None) -> dict[str, Any]:
        # the SDK call's kwargs
        raise NotImplementedError

    @abstractmethod
    async def _send(self, kwargs: dict[str, Any]) -> ResponseT:
        # the one SDK call generate() makes; _dispatch paces and retries it
        raise NotImplementedError

    @abstractmethod
    def _tokens_of(self, response: ResponseT) -> int:
        # what the rate gate charges against a tokens_per_minute quota
        raise NotImplementedError

    @abstractmethod
    def _parse(self, model: ModelT, response: ResponseT, output_schema: type[BaseModel] | None) -> LLMResponse:
        # the SDK's response object into an LLMResponse; raises LLMCallError for a refusal or an answer off the schema
        raise NotImplementedError

    async def _batch_api(self, task: Callable[[], Awaitable[T]]) -> T:
        # status checks, downloads and cancels can be repeated safely, so they get the usual retries. They bypass
        # the semaphore and the rate gates, which pace generation, not a provider's batch endpoints
        try:
            return await safe_call(task, retry_errors=self._retry_errors, retry_if=self._should_retry_on_error, retries=self._retries)
        except Exception as error:
            raise as_call_error(error, self.error_kind(error)) from error

    def _batch_size(self, request_id: str, body: dict[str, Any]) -> int:
        # the bytes one request adds to a provider batch, as the provider counts them against max_batch_bytes
        raise NotImplementedError(f"{self.provider_key.value} batches are not supported in this package")

    def _check_batch_id(self, request_id: str) -> None:
        # a provider with rules for its request ids raises ValueError here, before anything is sent
        return None

    async def _submit_batch(self, model: ModelT, requests: list[tuple[str, dict[str, Any]]]) -> str:
        # one provider batch from (id, _request kwargs) pairs; returns its batch id. Not retried: a create that
        # timed out may still have made the batch (see BatchRun._submit)
        raise NotImplementedError(f"{self.provider_key.value} batches are not supported in this package")

    async def _batch_ended(self, batch_id: str) -> bool:
        # True once the provider has finished with every request, however each one ended
        raise NotImplementedError(f"{self.provider_key.value} batches are not supported in this package")

    async def _batch_results(self, job: BatchJob) -> dict[str, ResponseT | LLMCallError]:
        # the SDK response per id, which BatchRun reads with _parse, or the error of a request that failed,
        # expired or was cancelled
        raise NotImplementedError(f"{self.provider_key.value} batches are not supported in this package")

    async def _cancel_batch(self, batch_id: str) -> None:
        raise NotImplementedError(f"{self.provider_key.value} batches are not supported in this package")

    async def embed(self, model_name: str, texts: list[str], task: EmbeddingTask) -> EmbeddingResponse:
        # task formatting, context guarding and batching are provider-independent; _embed_batch only
        # has to turn one ready-to-send batch into vectors
        if not self._embedding_models:
            raise NotImplementedError(f"{self.provider_key.value} embeddings are not supported in this package")
        model = self.resolve_embedding_model(model_name)
        if model.retired:
            raise ValueError(RETIRED_MESSAGE.format(name=model.model_name))
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

    async def _embed_batch(self, model: EmbeddingModel, texts: list[str]) -> EmbeddingBatch:
        # only reached by providers that register embedding models, and those override it
        raise NotImplementedError(f"{self.provider_key.value} embeddings are not supported in this package")
