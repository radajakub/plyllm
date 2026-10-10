"""Behaviour every provider client gets from `LLMClient` rather than implementing itself: the JSON-schema
check before a call, the error for a provider without embeddings, and the per-provider concurrency default."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from plybench.llm import (
    LLM,
    ClaudeProviderConfig,
    EmbeddingModelConfig,
    EmbeddingTask,
    GeminiProviderConfig,
    GrokProviderConfig,
    LLMCallOptions,
    LLMConfig,
    LLMMessage,
    MetacentrumProviderConfig,
    MistralProviderConfig,
    OpenAIProviderConfig,
    Provider,
)
from plybench.llm.providers.claude.client import ClaudeLLMClient
from plybench.llm.providers.gemini.client import GeminiLLMClient
from plybench.llm.providers.grok.client import GrokLLMClient
from plybench.llm.providers.metacentrum.client import MetacentrumLLMClient
from plybench.llm.providers.openai.client import OpenAILLMClient

_GENERATING = [Provider.OPENAI, Provider.GEMINI, Provider.GROK, Provider.CLAUDE, Provider.MISTRAL, Provider.METACENTRUM]


class _Answer(BaseModel):
    answer: int


def _llm(default_concurrency: int | None = None) -> LLM:
    # offline: building a client sends nothing, and every test here fails before a request would be sent
    return LLM(
        LLMConfig(
            openai=OpenAIProviderConfig(api_key="test"),
            gemini=GeminiProviderConfig(api_key="test"),
            grok=GrokProviderConfig(api_key="test"),
            claude=ClaudeProviderConfig(api_key="test"),
            mistral=MistralProviderConfig(api_key="test"),
            metacentrum=MetacentrumProviderConfig(api_key="test", base_url="https://offline.invalid/v1"),
            default_concurrency=default_concurrency,
        )
    )


@pytest.mark.parametrize("provider", _GENERATING)
def test_a_schema_is_refused_for_a_model_that_cannot_use_one(provider: Provider):
    llm = _llm()
    client = llm._provider_map[provider]
    model = next(model for model in client.get_available_models() if not model.retired)
    model.can_use_json_schema = False

    with pytest.raises(ValueError, match="does not support JSON schema"):
        asyncio.run(client.generate(model.model_name, LLMMessage.system("rules"), [LLMMessage.user("move")], LLMCallOptions(), _Answer))


@pytest.mark.parametrize("provider", [Provider.GROK, Provider.CLAUDE, Provider.MISTRAL, Provider.METACENTRUM])
def test_a_provider_without_embedding_models_says_embeddings_are_not_supported(provider: Provider):
    with pytest.raises(NotImplementedError, match=f"{provider.value} embeddings are not supported"):
        asyncio.run(_llm().embed(EmbeddingModelConfig(provider, "any"), ["hello"], EmbeddingTask.SEARCH_QUERY))


def test_each_provider_uses_its_own_concurrency_when_none_is_set():
    llm = _llm()

    limits = {provider: llm._provider_map[provider]._semaphore.concurrency for provider in _GENERATING}

    assert limits == {Provider.OPENAI: 10, Provider.GEMINI: 10, Provider.GROK: 10, Provider.CLAUDE: 10, Provider.MISTRAL: 10, Provider.METACENTRUM: 4}


def test_a_set_concurrency_overrides_every_provider_default():
    llm = _llm(default_concurrency=3)

    assert {llm._provider_map[provider]._semaphore.concurrency for provider in _GENERATING} == {3}


def test_aclose_closes_every_providers_connection_pool():
    llm = _llm()
    clients = llm._provider_map

    asyncio.run(llm.aclose())

    for client in (clients[Provider.OPENAI], clients[Provider.GROK], clients[Provider.CLAUDE], clients[Provider.METACENTRUM]):
        assert isinstance(client, OpenAILLMClient | GrokLLMClient | ClaudeLLMClient | MetacentrumLLMClient)
        assert client._client.is_closed(), client.provider_key
    gemini = clients[Provider.GEMINI]
    assert isinstance(gemini, GeminiLLMClient) and gemini._http_client is not None and gemini._http_client.is_closed
