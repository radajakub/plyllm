"""Why the provider stopped and how often the call was retried: each provider reads its own finish reason from the SDK's
response, generate() records it with the retries on the response (also on the response of an LLMCallError), and
`truncated` names a cut at the output limit whatever the provider calls it."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from anthropic.types import Message
from google.genai.types import Candidate, FinishReason, GenerateContentResponse
from mistralai.client.models import ChatCompletionChoice, ChatCompletionResponse
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.responses import Response
from openai.types.responses.response import IncompleteDetails
from pydantic import BaseModel

import plyllm.concurrency
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
    OllamaProviderConfig,
    OpenAIProviderConfig,
    Provider,
    VllmProviderConfig,
)
from plyllm.client import LLMClient


def _llm() -> LLM:
    # offline: no test here sends anything
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


def _responses(status: str, reason: str | None = None) -> Response:
    details = IncompleteDetails.model_construct(reason=reason) if reason is not None else None
    return Response.model_construct(status=status, incomplete_details=details)


def _chat(reason: str) -> ChatCompletion:
    return ChatCompletion.model_construct(choices=[Choice.model_construct(finish_reason=reason)])


# per provider, an SDK response cut at the output limit and what the provider calls that
_CUT: dict[Provider, tuple[Any, str]] = {
    Provider.OPENAI: (_responses("incomplete", "max_output_tokens"), "max_output_tokens"),
    Provider.GROK: (_responses("incomplete", "max_output_tokens"), "max_output_tokens"),
    Provider.METACENTRUM: (_responses("incomplete", "max_output_tokens"), "max_output_tokens"),
    Provider.CLAUDE: (Message.model_construct(stop_reason="max_tokens"), "max_tokens"),
    Provider.GEMINI: (GenerateContentResponse(candidates=[Candidate(finish_reason=FinishReason.MAX_TOKENS)]), "MAX_TOKENS"),
    Provider.MISTRAL: (ChatCompletionResponse.model_construct(choices=[ChatCompletionChoice.model_construct(finish_reason="length")]), "length"),
    Provider.VLLM: (_chat("length"), "length"),
    Provider.OLLAMA: (_chat("length"), "length"),
}

# per provider, an SDK response that ended normally and its finish reason
_DONE: dict[Provider, tuple[Any, str]] = {
    Provider.OPENAI: (_responses("completed"), "completed"),
    Provider.GROK: (_responses("completed"), "completed"),
    Provider.METACENTRUM: (_responses("completed"), "completed"),
    Provider.CLAUDE: (Message.model_construct(stop_reason="end_turn"), "end_turn"),
    Provider.GEMINI: (GenerateContentResponse(candidates=[Candidate(finish_reason=FinishReason.STOP)]), "STOP"),
    Provider.MISTRAL: (ChatCompletionResponse.model_construct(choices=[ChatCompletionChoice.model_construct(finish_reason="stop")]), "stop"),
    Provider.VLLM: (_chat("stop"), "stop"),
    Provider.OLLAMA: (_chat("stop"), "stop"),
}


@pytest.mark.parametrize("provider", list(_CUT))
def test_each_provider_reads_a_cut_answer_as_truncated(provider: Provider):
    client = _llm()._provider_map[provider]
    response, reason = _CUT[provider]

    finish_reason = client._finish_reason(response)

    assert finish_reason == reason
    assert LLMResponse(provider, "m", LLMTokens(), [], "", finish_reason=finish_reason).truncated


@pytest.mark.parametrize("provider", list(_DONE))
def test_each_provider_reads_a_finished_answer_as_not_truncated(provider: Provider):
    client = _llm()._provider_map[provider]
    response, reason = _DONE[provider]

    finish_reason = client._finish_reason(response)

    assert finish_reason == reason
    assert not LLMResponse(provider, "m", LLMTokens(), [], "", finish_reason=finish_reason).truncated


def test_a_response_without_a_finish_reason_is_not_truncated():
    assert not LLMResponse(Provider.OPENAI, "m", LLMTokens(), [], "").truncated


class _Flaky(Exception):
    pass


def _stubbed(client: LLMClient, monkeypatch: pytest.MonkeyPatch, failures: int, parse_error: bool = False) -> None:
    # the SDK call fails `failures` times with a retryable error, then answers; no wait between attempts
    attempts = 0

    async def send(kwargs: dict[str, Any]) -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= failures:
            raise _Flaky("try again")
        return "sdk response"

    def parse(model: LLMModel, response: str, output_schema: type[BaseModel] | None) -> LLMResponse:
        answer = LLMResponse.from_parts(client.provider_key, model.model_string, "e4", [], LLMTokens(), None)
        if parse_error:
            raise LLMCallError(FailureKind.UNPARSEABLE, "not the schema", answer)
        return answer

    async def no_wait(delay: float) -> None:
        return None

    monkeypatch.setattr(client, "_send", send)
    monkeypatch.setattr(client, "_tokens_of", lambda response: 0)
    monkeypatch.setattr(client, "_parse", parse)
    monkeypatch.setattr(client, "_finish_reason", lambda response: "length")
    monkeypatch.setattr(client, "_retry_errors", (_Flaky,))
    monkeypatch.setattr(client, "_should_retry_on_error", lambda error: True)
    monkeypatch.setattr(plyllm.concurrency.asyncio, "sleep", no_wait)


def _generate(client: LLMClient) -> LLMResponse:
    model = next(model for model in client.get_available_models() if not model.retired)
    return asyncio.run(client.generate(model.model_name, LLMMessage.system("rules"), [LLMMessage.user("move")], _options(model)))


def _options(model: LLMModel) -> LLMCallOptions:
    # options the model accepts; which does not matter here
    for options in (LLMCallOptions(), LLMCallOptions(thinking_enabled=True), LLMCallOptions(thinking_enabled=True, reasoning_effort="high")):
        try:
            model.validate(options)
        except ValueError:
            continue
        return options
    raise AssertionError(f"no options for {model.model_name}")


def test_generate_records_the_finish_reason_and_the_retries(monkeypatch: pytest.MonkeyPatch):
    client = _llm()._provider_map[Provider.MISTRAL]
    _stubbed(client, monkeypatch, failures=2)

    response = _generate(client)

    assert response.finish_reason == "length" and response.truncated
    assert response.retries == 2


def test_a_call_without_retries_records_none(monkeypatch: pytest.MonkeyPatch):
    client = _llm()._provider_map[Provider.MISTRAL]
    _stubbed(client, monkeypatch, failures=0)

    assert _generate(client).retries == 0


def test_an_answer_that_failed_records_its_finish_reason_and_retries(monkeypatch: pytest.MonkeyPatch):
    client = _llm()._provider_map[Provider.MISTRAL]
    _stubbed(client, monkeypatch, failures=1, parse_error=True)

    with pytest.raises(LLMCallError) as raised:
        _generate(client)

    response = raised.value.response
    assert response is not None and response.finish_reason == "length" and response.truncated and response.retries == 1
