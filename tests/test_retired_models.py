"""A model the provider no longer serves stays in its registry, marked `retired=True`: recorded results that
name it still resolve and cost, but a new call fails before anything is sent, with a message that says why
rather than the provider's 400."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from openai import AsyncOpenAI

from plybench.llm import (
    LLM,
    EmbeddingModelConfig,
    EmbeddingTask,
    LLMCallOptions,
    LLMConfig,
    LLMMessage,
    LLMTokens,
    MetacentrumProviderConfig,
    ModelConfig,
    OpenAIProviderConfig,
    Provider,
)
from plybench.llm.providers.metacentrum.client import MetacentrumLLMClient
from plybench.llm.providers.openai.client import OpenAILLMClient


class _MustNotBeCalled:
    async def create(self, **kwargs):
        raise AssertionError("a retired model reached the API")


def test_a_retired_model_is_refused_before_the_request_is_sent():
    client = MetacentrumLLMClient(cast(AsyncOpenAI, SimpleNamespace(responses=_MustNotBeCalled())))

    with pytest.raises(ValueError, match="glm-5.2 is retired"):
        asyncio.run(client.generate("glm-5.2", LLMMessage.system("rules"), [LLMMessage.user("move")], LLMCallOptions(thinking_enabled=True)))


def test_a_retired_model_still_resolves_and_costs_for_recorded_results():
    llm = LLM(LLMConfig(metacentrum=MetacentrumProviderConfig(api_key="test", base_url="https://offline.invalid/v1")))
    config = ModelConfig.from_string("metacentrum:glm-5.2:thinking_enabled=True")

    model = llm.resolve_model(Provider.METACENTRUM, "glm-5.2")

    assert model.retired and model.model_string == "glm-5.2"
    assert llm.calculate_cost(config, LLMTokens(input_tokens=100, output_tokens=50)) == 0.0


def test_a_retired_embedding_model_is_refused():
    llm = LLM(LLMConfig(openai=OpenAIProviderConfig(api_key="test")))
    client = llm._provider_map[Provider.OPENAI]
    assert isinstance(client, OpenAILLMClient)
    client.resolve_embedding_model("text-embedding-3-small").retired = True

    with pytest.raises(ValueError, match="text-embedding-3-small is retired"):
        asyncio.run(llm.embed(EmbeddingModelConfig(Provider.OPENAI, "text-embedding-3-small"), ["hello"], EmbeddingTask.SEARCH_QUERY))


def test_only_the_models_e_infra_dropped_are_retired():
    retired = {
        model.model_name
        for model in LLM(LLMConfig(metacentrum=MetacentrumProviderConfig(api_key="test", base_url="https://offline.invalid/v1"))).get_available_models(Provider.METACENTRUM)
        if model.retired
    }

    assert retired == {"deepseek-v4-flash", "deepseek-v3.2-thinking", "qwen-3.5-122b", "glm-5.2", "mistral-small-4"}
