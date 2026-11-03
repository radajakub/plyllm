"""The per-call seed: it reaches the request exactly where the provider's API takes one, and the response records
the seed the request carried, or None. A seed asked of a provider that takes none is a warning."""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

import pytest
from pydantic import BaseModel

from plyllm import (
    LLM,
    ClaudeProviderConfig,
    FailureKind,
    GeminiProviderConfig,
    GrokProviderConfig,
    LLMCallError,
    LLMCallOptions,
    LLMConfig,
    LLMMessage,
    LLMModel,
    LLMResponse,
    LLMTokens,
    MetacentrumProviderConfig,
    MistralProviderConfig,
    ModelConfig,
    OllamaProviderConfig,
    OpenAIProviderConfig,
    Provider,
    VllmProviderConfig,
)
from plyllm.client import LLMClient

# where each provider that takes a seed puts it in the request
_SEEDED = {
    Provider.GEMINI: lambda kwargs: kwargs["config"].seed,
    Provider.MISTRAL: lambda kwargs: kwargs.get("random_seed"),
    Provider.VLLM: lambda kwargs: kwargs.get("seed"),
    Provider.OLLAMA: lambda kwargs: kwargs.get("seed"),
}
_UNSEEDED = [Provider.OPENAI, Provider.GROK, Provider.CLAUDE, Provider.METACENTRUM]
# the keys a seed could take in any of the SDKs' kwargs
_SEED_KEYS = ("seed", "random_seed")


class _Answer(BaseModel):
    answer: int


def _llm() -> LLM:
    # offline: _send is replaced below, so nothing is sent
    return LLM(
        LLMConfig(
            openai=OpenAIProviderConfig(api_key="test"),
            gemini=GeminiProviderConfig(api_key="test"),
            grok=GrokProviderConfig(api_key="test"),
            claude=ClaudeProviderConfig(api_key="test"),
            mistral=MistralProviderConfig(api_key="test"),
            metacentrum=MetacentrumProviderConfig(api_key="test", base_url="https://offline.invalid/v1"),
            vllm=VllmProviderConfig(api_key="test", base_url="https://offline.invalid/v1"),
            ollama=OllamaProviderConfig(base_url="https://offline.invalid/v1"),
        )
    )


def _callable(client: LLMClient) -> tuple[LLMModel, LLMCallOptions]:
    # any served model with options it accepts; which one does not matter for where the seed goes
    candidates = [LLMCallOptions(), LLMCallOptions(thinking_enabled=True), LLMCallOptions(thinking_enabled=True, reasoning_effort="high")]
    for model in client.get_available_models():
        if model.retired:
            continue
        for options in candidates:
            try:
                model.validate(options)
            except ValueError:
                continue
            return model, options
    raise AssertionError(f"no callable model for {client.provider_key.value}")


def _captured(client: LLMClient, monkeypatch: pytest.MonkeyPatch, parse_error: bool = False) -> list[dict[str, Any]]:
    # replaces the SDK call: the kwargs are kept, and the answer is a fixed one, or an unparseable one
    sent: list[dict[str, Any]] = []

    async def send(kwargs: dict[str, Any]) -> None:
        sent.append(kwargs)

    def parse(model: LLMModel, response: None, output_schema: type[BaseModel] | None) -> LLMResponse:
        answer = LLMResponse.from_parts(client.provider_key, model.model_string, "e4", [], LLMTokens(), None)
        if parse_error:
            raise LLMCallError(FailureKind.UNPARSEABLE, "not the schema", answer)
        return answer

    monkeypatch.setattr(client, "_send", send)
    monkeypatch.setattr(client, "_tokens_of", lambda response: 0)
    monkeypatch.setattr(client, "_parse", parse)
    return sent


def _generate(client: LLMClient, seed: int | None) -> LLMResponse:
    model, options = _callable(client)
    return asyncio.run(client.generate(model.model_name, LLMMessage.system("rules"), [LLMMessage.user("move")], options, None, seed))


@pytest.mark.parametrize("provider", list(_SEEDED))
def test_a_provider_that_takes_a_seed_sends_it_and_records_it(provider: Provider, monkeypatch: pytest.MonkeyPatch):
    client = _llm()._provider_map[provider]
    sent = _captured(client, monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        response = _generate(client, 7)

    assert _SEEDED[provider](sent[0]) == 7
    assert response.seed == 7


@pytest.mark.parametrize("provider", _UNSEEDED)
def test_a_provider_without_a_seed_warns_sends_none_and_records_none(provider: Provider, monkeypatch: pytest.MonkeyPatch):
    client = _llm()._provider_map[provider]
    sent = _captured(client, monkeypatch)

    with pytest.warns(UserWarning, match=f"{provider.value} takes no sampling seed"):
        response = _generate(client, 7)

    assert not any(key in sent[0] for key in _SEED_KEYS)
    assert response.seed is None


@pytest.mark.parametrize("provider", [*_SEEDED, *_UNSEEDED])
def test_no_seed_asked_sends_none_records_none_and_does_not_warn(provider: Provider, monkeypatch: pytest.MonkeyPatch):
    client = _llm()._provider_map[provider]
    sent = _captured(client, monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        response = _generate(client, None)

    # a provider that takes a seed has its field empty; the others have no field
    assert _SEEDED[provider](sent[0]) is None if provider in _SEEDED else not any(key in sent[0] for key in _SEED_KEYS)
    assert response.seed is None


def test_an_answer_that_failed_records_its_seed(monkeypatch: pytest.MonkeyPatch):
    client = _llm()._provider_map[Provider.MISTRAL]
    _captured(client, monkeypatch, parse_error=True)

    with pytest.raises(LLMCallError) as raised:
        _generate(client, 7)

    assert raised.value.response is not None and raised.value.response.seed == 7


def test_the_router_passes_the_seed_to_the_client(monkeypatch: pytest.MonkeyPatch):
    llm = _llm()
    client = llm._provider_map[Provider.MISTRAL]
    sent = _captured(client, monkeypatch)
    model, options = _callable(client)

    response = asyncio.run(llm.generate(ModelConfig(Provider.MISTRAL, model.model_name, options), LLMMessage.system("rules"), [LLMMessage.user("move")], seed=7))

    assert sent[0]["random_seed"] == 7 and response.seed == 7
