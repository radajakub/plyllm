from __future__ import annotations

import importlib

from pydantic import BaseModel

from omniplay.llm.client import LLMClient
from omniplay.llm.llm_config import LLMConfig
from omniplay.llm.message import LLMMessage
from omniplay.llm.model import LLMModel
from omniplay.llm.model_config import ModelConfig
from omniplay.llm.providers.providers import Provider
from omniplay.llm.response import EmbeddingResponse, LLMResponse
from omniplay.llm.tokens import LLMTokens

# each provider SDK is an optional extra; wire up only the ones whose dependencies are installed
_CLIENT_MODULES = (
    ("omniplay.llm.providers.openai.client", "OpenAILLMClient"),
    ("omniplay.llm.providers.gemini.client", "GeminiLLMClient"),
    ("omniplay.llm.providers.metacentrum.client", "MetacentrumLLMClient"),
    ("omniplay.llm.providers.huggingface.client", "HuggingFaceLLMClient"),
)


def _client_builders() -> list[type[LLMClient]]:
    builders: list[type[LLMClient]] = []
    for module_path, class_name in _CLIENT_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # provider extra not installed -- skip it
            continue
        builders.append(getattr(module, class_name))
    return builders


class LLM:
    def __init__(self, config: LLMConfig) -> None:
        self._provider_map: dict[Provider, LLMClient] = {}
        for builder in _client_builders():
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

    def resolve_model(self, provider: Provider, model_name: str) -> LLMModel:
        return self._route(provider).resolve_model(model_name)

    def calculate_cost(self, model_config: ModelConfig, tokens: LLMTokens) -> float:
        return self._route(model_config.provider).calculate_cost(model_config.model_name, tokens)

    def set_concurrency(self, provider: Provider, concurrency: int | None) -> None:
        self._route(provider).set_concurrency(concurrency)

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

    async def embed(self, provider: Provider, model_name: str, texts: list[str]) -> EmbeddingResponse:
        return await self._route(provider).embed(model_name, texts)
