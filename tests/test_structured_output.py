"""One contract for answers that arrive but cannot be used.

Every provider hands its raw answer to `LLMClient._answer`, which checks it against the schema the same
way for all of them. An answer that does not fit fails as UNPARSEABLE, a declined request as REFUSAL, and
both carry the response that did arrive, so its text and its cost are not lost with the exception. The
SDKs' own parse helpers validated inside the call instead: they lost both, and a Claude refusal under a
schema surfaced as a schema failure because the SDK choked on the refusal text first.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from anthropic import AsyncAnthropic, transform_schema
from google.genai.client import AsyncClient
from google.genai.types import BlockedReason, Candidate, FinishReason, GenerateContentResponse, GenerateContentResponsePromptFeedback
from openai import AsyncOpenAI
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import BaseModel

from plybench.llm import ClaudeProviderConfig, FailureKind, LLMCallError, LLMCallOptions, LLMConfig, LLMMessage, LLMResponse, LLMTokens
from plybench.llm.llm_config import DEFAULT_TIMEOUT
from plybench.llm.providers.claude.client import ClaudeLLMClient
from plybench.llm.providers.gemini.client import GeminiLLMClient
from plybench.llm.providers.openai.client import OpenAILLMClient
from plybench.llm.providers.openai.models import openai_models


class _Move(BaseModel):
    move: int


_TOKENS = LLMTokens(input_tokens=10, output_tokens=5)


# --- the check itself ---------------------------------------------------------------------------------


def _check(output_text: str) -> LLMResponse:
    # the check lives on the base client; any provider's instance exercises the same code
    client = OpenAILLMClient.__new__(OpenAILLMClient)
    model = next(model for model in openai_models() if model.model_name == "gpt-5.4")
    return client._answer(model, output_text, ["thought"], _TOKENS, _Move)


def test_a_fitting_answer_is_stored_in_one_normalised_form():
    response = _check('{ "move" :4 }')

    assert response.output_text == '{"move":4}'
    assert response.structured_output_type is _Move
    assert response.resolve_structured_output(_Move) == _Move(move=4)
    assert response.reasoning == ["thought"]


def test_an_answer_that_does_not_fit_fails_with_the_raw_answer_attached():
    with pytest.raises(LLMCallError) as caught:
        _check('{"move": "centre"}')

    raw = caught.value.response
    assert caught.value.kind is FailureKind.UNPARSEABLE
    assert raw is not None
    assert raw.output_text == '{"move": "centre"}' and raw.reasoning == ["thought"] and raw.tokens == _TOKENS
    # the raw text does not conform, so it does not claim the schema
    assert raw.structured_output_type is None


# --- the requests are unchanged from what the SDKs' parse helpers sent ---------------------------------


class _Recorder:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(200, json=self._body)


def _openai_body(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5.4",
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": content}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
    }


def _openai_text(text: str) -> dict[str, Any]:
    return _openai_body([{"type": "output_text", "text": text, "annotations": []}])


def _openai_sdk(recorder: _Recorder) -> AsyncOpenAI:
    return AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)))


def test_openai_sends_the_schema_responses_parse_would_send():
    """Also pins the private `type_to_text_format_param` import: an SDK upgrade that moves or changes it
    fails here rather than in a run."""
    recorder = _Recorder(_openai_text('{"move": 4}'))
    sdk = _openai_sdk(recorder)

    async def both() -> None:
        await sdk.responses.parse(model="gpt-5.4", input="go", text_format=_Move)
        await sdk.responses.create(model="gpt-5.4", input="go", text={"format": type_to_text_format_param(_Move)})

    asyncio.run(both())

    assert recorder.requests[0] == recorder.requests[1]


def _anthropic_body(content: list[dict[str, Any]], stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _anthropic_sdk(recorder: _Recorder) -> AsyncAnthropic:
    # an explicit timeout, as ClaudeLLMClient.build() sets: with the SDK's default one, a non-streamed request
    # whose max_tokens suggests over ten minutes is refused before it is sent
    return AsyncAnthropic(api_key="test", timeout=DEFAULT_TIMEOUT, http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)))


def test_claude_sends_the_schema_messages_parse_would_send():
    recorder = _Recorder(_anthropic_body([{"type": "text", "text": '{"move": 4}'}]))
    sdk = _anthropic_sdk(recorder)
    common: dict[str, Any] = dict(model="claude-opus-5", max_tokens=100, messages=[{"role": "user", "content": "go"}])

    async def both() -> None:
        await sdk.messages.parse(**common, output_format=_Move, output_config={"effort": "high"})
        await sdk.messages.create(**common, output_config={"effort": "high", "format": {"type": "json_schema", "schema": transform_schema(_Move.model_json_schema())}})

    asyncio.run(both())

    assert recorder.requests[0] == recorder.requests[1]


# --- each provider raises the shared kinds, with the answer attached -----------------------------------


def _generate(client, model: str, options: LLMCallOptions | None = None) -> LLMResponse:
    return asyncio.run(client.generate(model, LLMMessage.system("rules"), [LLMMessage.user("your move")], options or LLMCallOptions(), output_schema=_Move))


def test_openai_answer_that_does_not_fit_keeps_its_text_and_tokens():
    client = OpenAILLMClient(_openai_sdk(_Recorder(_openai_text('{"move": "centre"}'))), concurrency=1)

    with pytest.raises(LLMCallError) as caught:
        _generate(client, "gpt-5.4")

    assert caught.value.kind is FailureKind.UNPARSEABLE
    assert caught.value.response is not None
    assert caught.value.response.output_text == '{"move": "centre"}' and caught.value.response.tokens.output_tokens == 5


def test_openai_refusal_is_a_refusal_not_a_schema_failure():
    client = OpenAILLMClient(_openai_sdk(_Recorder(_openai_body([{"type": "refusal", "refusal": "I can't help with that."}]))), concurrency=1)

    with pytest.raises(LLMCallError) as caught:
        _generate(client, "gpt-5.4")

    assert caught.value.kind is FailureKind.REFUSAL
    assert caught.value.response is not None and caught.value.response.output_text == "I can't help with that."


def test_claude_refusal_under_a_schema_is_a_refusal():
    """The SDK's parse helper validated the refusal text against the schema and raised before stop_reason
    could be read, so the judge saw these as schema failures."""
    body = _anthropic_body([{"type": "text", "text": "I won't do that."}], stop_reason="refusal")
    client = ClaudeLLMClient(_anthropic_sdk(_Recorder(body)), concurrency=1)

    with pytest.raises(LLMCallError) as caught:
        _generate(client, "claude-opus-5", LLMCallOptions(thinking_enabled=True, reasoning_effort="high"))

    assert caught.value.kind is FailureKind.REFUSAL
    assert caught.value.response is not None and caught.value.response.output_text == "I won't do that."


def test_claude_merges_the_schema_into_the_effort_setting():
    recorder = _Recorder(_anthropic_body([{"type": "text", "text": '{"move": 4}'}]))
    client = ClaudeLLMClient(_anthropic_sdk(recorder), concurrency=1)

    response = _generate(client, "claude-opus-5", LLMCallOptions(thinking_enabled=True, reasoning_effort="high"))

    sent = recorder.requests[0]["output_config"]
    assert sent["effort"] == "high" and sent["format"]["type"] == "json_schema"
    assert response.resolve_structured_output(_Move) == _Move(move=4)


def _gemini(response: GenerateContentResponse) -> GeminiLLMClient:
    async def generate_content(**_kwargs) -> GenerateContentResponse:
        return response

    return GeminiLLMClient(cast(AsyncClient, SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))), concurrency=1)


@pytest.mark.parametrize(
    "response",
    [
        GenerateContentResponse(prompt_feedback=GenerateContentResponsePromptFeedback(block_reason=BlockedReason.SAFETY)),
        GenerateContentResponse(candidates=[Candidate(finish_reason=FinishReason.SAFETY)]),
    ],
)
def test_gemini_blocked_answer_is_a_refusal(response: GenerateContentResponse):
    # Gemini has no refusal message: a declined request is a blocked prompt or an answer stopped by a filter
    with pytest.raises(LLMCallError) as caught:
        _generate(_gemini(response), "gemini-3.5-flash", LLMCallOptions(thinking_enabled=True))

    assert caught.value.kind is FailureKind.REFUSAL
    assert caught.value.response is not None


def test_a_built_claude_client_sends_long_requests_without_streaming(monkeypatch):
    """The SDK refuses a non-streamed request whose max_tokens suggests over ten minutes, unless the client
    has its own timeout. `build()` always sets one; this keeps it that way."""
    recorder = _Recorder(_anthropic_body([{"type": "text", "text": '{"move": 4}'}]))
    real = AsyncAnthropic

    def with_mock_transport(**kwargs):
        return real(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)))

    monkeypatch.setattr("plybench.llm.providers.claude.client.AsyncAnthropic", with_mock_transport)
    client = ClaudeLLMClient.build(LLMConfig(claude=ClaudeProviderConfig(api_key="test")))
    assert client is not None

    # Opus 5.5 defaults to max_tokens=32000, well over what the SDK allows unstreamed by default
    _generate(client, "claude-opus-5.5", LLMCallOptions(thinking_enabled=True, reasoning_effort="max"))

    assert recorder.requests[0]["max_tokens"] == 32000
