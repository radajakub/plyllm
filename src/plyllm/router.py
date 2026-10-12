from __future__ import annotations

import asyncio
import importlib
from collections.abc import Sequence

from pydantic import BaseModel

from plyllm.batch import DEFAULT_POLL_INTERVAL, BatchRequest, BatchStore
from plyllm.batch_run import BatchRun, cancel_batches
from plyllm.client import LLMClient
from plyllm.decision import DecisionQuestion
from plyllm.llm_config import LLMConfig
from plyllm.message import LLMMessage
from plyllm.model import DecisionModel, EmbeddingModel, EmbeddingTask, LLMModel
from plyllm.model_config import DecisionModelConfig, EmbeddingModelConfig, ModelConfig
from plyllm.providers.providers import Provider
from plyllm.rate_limit import ModelLimits
from plyllm.response import DecisionResponse, EmbeddingResponse, LLMResponse
from plyllm.tokens import EmbeddingTokens, LLMTokens

# Each provider SDK is an optional extra. Import a provider only when it is configured.
_CLIENT_MODULES = (
    (Provider.OPENAI, "plyllm.providers.openai.client", "OpenAILLMClient"),
    (Provider.GEMINI, "plyllm.providers.gemini.client", "GeminiLLMClient"),
    (Provider.GROK, "plyllm.providers.grok.client", "GrokLLMClient"),
    (Provider.CLAUDE, "plyllm.providers.claude.client", "ClaudeLLMClient"),
    (Provider.MISTRAL, "plyllm.providers.mistral.client", "MistralLLMClient"),
    (Provider.METACENTRUM, "plyllm.providers.metacentrum.client", "MetacentrumLLMClient"),
    (Provider.VLLM, "plyllm.providers.vllm.client", "VllmLLMClient"),
    (Provider.OLLAMA, "plyllm.providers.ollama.client", "OllamaLLMClient"),
    (Provider.HUGGINGFACE, "plyllm.providers.huggingface.client", "HuggingFaceLLMClient"),
)


def _client_builders(config: LLMConfig) -> list[type[LLMClient]]:
    builders: list[type[LLMClient]] = []
    for provider, module_path, class_name in _CLIENT_MODULES:
        if getattr(config, provider.value) is None:
            continue
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # Preserve the optional-extra behavior: an unavailable provider is skipped.
            continue
        builders.append(getattr(module, class_name))
    return builders


class LLM:
    def __init__(self, config: LLMConfig) -> None:
        self._provider_map: dict[Provider, LLMClient] = {}
        for builder in _client_builders(config):
            client = builder.build(config)
            if client is not None:
                self._provider_map[client.provider_key] = client

    def _route(self, provider: Provider) -> LLMClient:
        client = self._provider_map.get(provider)
        if client is None:
            raise ValueError(f"Provider {provider.value} is not configured (missing credentials). Available providers: {[p.value for p in self._provider_map]}")
        return client

    def bootstrap(self) -> None:
        # let providers that need to download/verify local resources do so (no-op for remote ones)
        for client in self._provider_map.values():
            client.bootstrap()

    @property
    def available_providers(self) -> list[Provider]:
        return list(self._provider_map.keys())

    def get_available_models(self, provider: Provider) -> list[LLMModel]:
        return self._route(provider).get_available_models()

    def get_available_embedding_models(self, provider: Provider) -> list[EmbeddingModel]:
        return self._route(provider).get_available_embedding_models()

    def get_available_decision_models(self, provider: Provider) -> list[DecisionModel]:
        return self._route(provider).get_available_decision_models()

    async def aclose(self) -> None:
        # closes every provider's connection pool; await it at the end of the asyncio.run() that used them
        await asyncio.gather(*(client.aclose() for client in self._provider_map.values()))

    async def served_models(self, provider: Provider) -> set[str]:
        return await self._route(provider).served_models()

    def resolve_model(self, provider: Provider, model_name: str) -> LLMModel:
        return self._route(provider).resolve_model(model_name)

    def resolve_embedding_model(self, provider: Provider, model_name: str) -> EmbeddingModel:
        return self._route(provider).resolve_embedding_model(model_name)

    def resolve_decision_model(self, provider: Provider, model_name: str) -> DecisionModel:
        return self._route(provider).resolve_decision_model(model_name)

    def calculate_cost(self, model_config: ModelConfig, tokens: LLMTokens, batch: bool = False) -> float:
        # batch=True prices the tokens at the provider's Batch API rate
        return self._route(model_config.provider).calculate_cost(model_config.model_name, tokens, batch)

    def calculate_embedding_cost(self, model_config: EmbeddingModelConfig, tokens: EmbeddingTokens) -> float:
        return self._route(model_config.provider).calculate_embedding_cost(model_config.model_name, tokens)

    def calculate_decision_cost(self, model_config: DecisionModelConfig, tokens: LLMTokens) -> float:
        return self._route(model_config.provider).calculate_decision_cost(model_config.model_name, tokens)

    def set_concurrency(self, provider: Provider, concurrency: int | None) -> None:
        self._route(provider).set_concurrency(concurrency)

    def set_model_limits(self, provider: Provider, model_name: str, limits: ModelLimits | None) -> None:
        self._route(provider).set_model_limits(model_name, limits)

    def set_embedding_model_limits(self, provider: Provider, model_name: str, limits: ModelLimits | None) -> None:
        self._route(provider).set_embedding_model_limits(model_name, limits)

    def set_decision_model_limits(self, provider: Provider, model_name: str, limits: ModelLimits | None) -> None:
        self._route(provider).set_decision_model_limits(model_name, limits)

    async def generate(
        self,
        model_config: ModelConfig,
        system: LLMMessage,
        messages: list[LLMMessage],
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        client = self._route(model_config.provider)
        return await client.generate(
            model_config.model_name,
            system,
            messages,
            model_config.options,
            output_schema,
        )

    def batch(
        self,
        model_config: ModelConfig,
        requests: Sequence[BatchRequest],
        store: BatchStore,
        output_schema: type[BaseModel] | None = None,
        wait: bool = False,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> BatchRun:
        return BatchRun(self._route(model_config.provider), model_config.model_name, model_config.options, requests, store, output_schema, wait, poll_interval)

    async def cancel_batches(self, provider: Provider, store: BatchStore) -> None:
        await cancel_batches(self._route(provider), store)

    async def embed(self, model_config: EmbeddingModelConfig, texts: list[str], task: EmbeddingTask) -> EmbeddingResponse:
        return await self._route(model_config.provider).embed(model_config.model_name, texts, task)

    async def decide(self, model_config: DecisionModelConfig, input: str, questions: Sequence[DecisionQuestion]) -> DecisionResponse:
        return await self._route(model_config.provider).decide(model_config.model_name, input, questions)
