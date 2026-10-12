"""The ollama provider against a fake SDK: the alias apart from the tag, the sampling Ollama's /v1 takes, and
thinking switched through reasoning_effort. Reading the answers is the open-weight client's, tested with vllm."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from plyllm import LLM, LLMCallOptions, LLMConfig, LLMMessage, ModelConfig, OllamaProviderConfig, Provider
from plyllm.providers.ollama.client import OllamaLLMClient
from plyllm.providers.ollama.models import OllamaLLMModel, ollama_models


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> ChatCompletion:
        self.kwargs = kwargs
        return ChatCompletion.model_validate(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gemma4:e4b",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "4", "reasoning": "two and two"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )


def _sent(options: LLMCallOptions) -> dict[str, Any]:
    completions = _FakeCompletions()
    client = OllamaLLMClient(cast(AsyncOpenAI, SimpleNamespace(chat=SimpleNamespace(completions=completions))))
    asyncio.run(client.generate("gemma-4-e4b", LLMMessage.system("Answer."), [LLMMessage.user("2+2?")], options))
    return completions.kwargs


def test_from_env_needs_only_the_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "")
    monkeypatch.setenv("OLLAMA_API_KEY", "")
    assert LLMConfig.from_env().ollama is None

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    assert LLMConfig.from_env().ollama == OllamaProviderConfig(base_url="http://127.0.0.1:11434/v1", api_key="ollama")


def test_the_router_builds_the_ollama_client():
    llm = LLM(LLMConfig(ollama=OllamaProviderConfig(base_url="http://127.0.0.1:1/v1")))

    assert isinstance(llm._provider_map[Provider.OLLAMA], OllamaLLMClient)


def test_the_alias_has_no_colon_so_a_config_string_can_name_it():
    # a tag such as gemma4:e4b would split the provider:model:options string
    config = ModelConfig.from_string("ollama:gemma-4-e4b:thinking_enabled=True")

    assert (config.provider, config.model_name, config.options.thinking_enabled) == (Provider.OLLAMA, "gemma-4-e4b", True)
    assert all(":" not in model.model_name for model in ollama_models())


def test_the_tag_is_sent_and_thinking_off_is_effort_none():
    assert _sent(LLMCallOptions()) == {
        "model": "gemma4:e4b",
        "messages": [{"role": "system", "content": "Answer."}, {"role": "user", "content": "2+2?"}],
        "extra_body": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0, "frequency_penalty": 0.0, "reasoning_effort": "none"},
    }


def test_thinking_on_without_levels_is_any_effort():
    body = _sent(LLMCallOptions(thinking_enabled=True, temperature=0.0, max_tokens=64))

    assert body["extra_body"]["reasoning_effort"] == "medium"
    assert body["extra_body"]["temperature"] == 0.0 and body["max_tokens"] == 64
    assert "chat_template_kwargs" not in body["extra_body"]


def test_a_model_with_levels_takes_only_the_callers():
    model = OllamaLLMModel("m", "m:tag", sampling_thinking={"temperature": 1.0}, thinking=True, thinking_only=True, supported_reasoning=frozenset({"low", "high"}))

    with pytest.raises(ValueError, match="does not reason without a reasoning_effort"):
        model.extract_params(LLMCallOptions(thinking_enabled=True))
    assert model.extract_params(LLMCallOptions(thinking_enabled=True, reasoning_effort="low"))["extra_body"]["reasoning_effort"] == "low"


def test_sampling_the_endpoint_would_ignore_is_refused():
    with pytest.raises(ValueError, match=r"ignores \['top_k'\]"):
        OllamaLLMModel("m", "m:tag", sampling_no_thinking={"temperature": 1.0, "top_k": 64})
