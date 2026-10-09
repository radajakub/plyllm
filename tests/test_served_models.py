"""`served_models` lists what a provider's endpoint serves, so a retired or mistyped model string is caught
before a run. Each provider reads its own SDK's listing; the live tests check the result against the
registries, these check the reading against fake SDKs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic
from google.genai.client import AsyncClient
from mistralai.client import Mistral
from mistralai.client.models import BaseModelCard, FTModelCard
from openai import AsyncOpenAI

from plybench.llm import LLM, HuggingFaceProviderConfig, LLMConfig, OpenAIProviderConfig, Provider
from plybench.llm.providers.claude.client import ClaudeLLMClient
from plybench.llm.providers.gemini.client import GeminiLLMClient
from plybench.llm.providers.grok.client import GrokLLMClient
from plybench.llm.providers.huggingface.client import HuggingFaceLLMClient
from plybench.llm.providers.metacentrum.client import MetacentrumLLMClient
from plybench.llm.providers.mistral.client import MistralLLMClient
from plybench.llm.providers.openai.client import OpenAILLMClient


class _Pager:
    # what the SDKs' list calls return: an object that pages through results on `async for`
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    async def __aiter__(self) -> AsyncIterator[Any]:
        for item in self._items:
            yield item


def _listing(*ids: str) -> SimpleNamespace:
    return SimpleNamespace(models=SimpleNamespace(list=lambda: _Pager([SimpleNamespace(id=model_id) for model_id in ids])))


@pytest.mark.parametrize(
    "build",
    [
        lambda sdk: OpenAILLMClient(cast(AsyncOpenAI, sdk)),
        lambda sdk: GrokLLMClient(cast(AsyncOpenAI, sdk)),
        lambda sdk: MetacentrumLLMClient(cast(AsyncOpenAI, sdk)),
        lambda sdk: ClaudeLLMClient(cast(AsyncAnthropic, sdk)),
    ],
    ids=["openai", "grok", "metacentrum", "claude"],
)
def test_openai_and_anthropic_sdks_list_model_ids(build):
    client = build(_listing("model-a", "model-b"))

    assert asyncio.run(client.served_models()) == {"model-a", "model-b"}


def test_gemini_strips_the_models_prefix():
    async def list_models() -> _Pager:
        return _Pager([SimpleNamespace(name="models/gemini-3.5-flash"), SimpleNamespace(name=None)])

    client = GeminiLLMClient(cast(AsyncClient, SimpleNamespace(models=SimpleNamespace(list=list_models))))

    assert asyncio.run(client.served_models()) == {"gemini-3.5-flash"}


def test_mistral_lists_ids_and_aliases_and_skips_unknown_cards():
    cards = [
        BaseModelCard.model_construct(id="mistral-small-2603", aliases=["mistral-small-latest"]),
        FTModelCard.model_construct(id="ft:mistral-small:abc", aliases=None),
        SimpleNamespace(type="something-new"),  # a card type the SDK does not model: no id to read
    ]

    async def list_async() -> SimpleNamespace:
        return SimpleNamespace(data=cards)

    client = MistralLLMClient(cast(Mistral, SimpleNamespace(models=SimpleNamespace(list_async=list_async))))

    assert asyncio.run(client.served_models()) == {"mistral-small-2603", "mistral-small-latest", "ft:mistral-small:abc"}


def test_local_models_have_no_endpoint_to_list():
    client = HuggingFaceLLMClient(HuggingFaceProviderConfig(models=()))

    with pytest.raises(NotImplementedError):
        asyncio.run(client.served_models())


def test_the_router_asks_the_provider():
    llm = LLM(LLMConfig(openai=OpenAIProviderConfig(api_key="test")))
    llm._provider_map[Provider.OPENAI]._client = _listing("gpt-6-luna")  # type: ignore[attr-defined]

    assert asyncio.run(llm.served_models(Provider.OPENAI)) == {"gpt-6-luna"}
