"""One contract for answers that arrive but cannot be used.

Every provider hands its raw answer to `LLMClient._answer`, which checks it against the schema the same
way for all of them. An answer that does not fit fails as UNPARSEABLE, a declined request as REFUSAL, and
both carry the response that did arrive, so its text and its cost are not lost with the exception. The
SDKs' own parse helpers validated inside the call instead: they lost both, and a Claude refusal under a
schema surfaced as a schema failure because the SDK choked on the refusal text first.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any, cast

import httpx
import httpx2
import pytest
from anthropic import AsyncAnthropic, transform_schema
from google import genai
from google.genai.client import AsyncClient
from google.genai.types import (
    BlockedReason,
    Candidate,
    Content,
    FinishReason,
    GenerateContentResponse,
    GenerateContentResponsePromptFeedback,
    GenerateContentResponseUsageMetadata,
    HttpOptions,
    Part,
)
from openai import AsyncOpenAI
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import BaseModel

from plyllm import ClaudeProviderConfig, FailureKind, LLMCallError, LLMCallOptions, LLMConfig, LLMMessage, LLMResponse, LLMTokens
from plyllm.llm_config import DEFAULT_TIMEOUT
from plyllm.providers.claude.client import ClaudeLLMClient
from plyllm.providers.gemini.client import _MAX_HOPS, GeminiLLMClient
from plyllm.providers.openai.client import OpenAILLMClient
from plyllm.providers.openai.models import openai_models


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

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return httpx2.Response(200, json=self._body)


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
    return AsyncOpenAI(api_key="test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)))


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
    return AsyncAnthropic(api_key="test", timeout=DEFAULT_TIMEOUT, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)))


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


def test_gemini_blocked_answer_is_a_refusal():
    # Gemini has no refusal message: a declined request is an answer stopped by a filter
    response = GenerateContentResponse(candidates=[Candidate(finish_reason=FinishReason.SAFETY)])
    with pytest.raises(LLMCallError) as caught:
        _generate(_gemini(response), "gemini-3.6-flash", LLMCallOptions(thinking_enabled=True))

    assert caught.value.kind is FailureKind.REFUSAL
    assert caught.value.response is not None


@pytest.mark.parametrize("reason", [BlockedReason.SAFETY, BlockedReason.OTHER])
def test_gemini_blocked_prompt_is_kept_apart_from_a_refusal(reason: BlockedReason):
    """The same request was seen blocked once and answered another time, so a caller must be able to tell a
    blocked prompt from the model declining, to send it again."""
    response = GenerateContentResponse(
        prompt_feedback=GenerateContentResponsePromptFeedback(block_reason=reason), usage_metadata=GenerateContentResponseUsageMetadata(prompt_token_count=368)
    )
    with pytest.raises(LLMCallError) as caught:
        _generate(_gemini(response), "gemini-3.6-flash", LLMCallOptions(thinking_enabled=True))

    assert caught.value.kind is FailureKind.PROMPT_BLOCKED
    assert f"prompt blocked ({reason.value})" in str(caught.value)
    assert caught.value.response is not None and caught.value.response.tokens.input_tokens == 368


def _hop(parts: list[Part], finish: FinishReason, token: bytes | None = None) -> GenerateContentResponse:
    candidate = Candidate(content=Content(role="model", parts=parts), finish_reason=finish, continuation_token=token)
    return GenerateContentResponse(
        candidates=[candidate], usage_metadata=GenerateContentResponseUsageMetadata(prompt_token_count=100, candidates_token_count=10, thoughts_token_count=5)
    )


def _gemini_hops(hops: list[GenerateContentResponse], configs: list[Any]) -> GeminiLLMClient:
    # answers the n-th request with the n-th hop, the last one for any request beyond them
    async def generate_content(**kwargs) -> GenerateContentResponse:
        configs.append(kwargs["config"])
        return hops[min(len(configs), len(hops)) - 1]

    return GeminiLLMClient(cast(AsyncClient, SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))), concurrency=1)


def test_gemini_answer_split_across_hops_is_joined_and_every_hop_is_paid_for():
    """Gemini stops a long answer with CONTINUATION and a token; plyllm asks again with the token, joins the
    hops, and counts each hop's tokens, since each one bills the prompt again."""
    hops = [
        _hop([Part(text="weighing the centre", thought=True), Part(text='{"mo')], FinishReason.CONTINUATION, b"resume"),
        _hop([Part(text='ve": 4}')], FinishReason.STOP),
    ]
    configs: list[Any] = []

    response = _generate(_gemini_hops(hops, configs), "gemini-3.6-flash", LLMCallOptions(thinking_enabled=True))

    assert response.resolve_structured_output(_Move) == _Move(move=4)
    assert response.reasoning == ["weighing the centre"]
    assert response.tokens == LLMTokens(input_tokens=200, output_tokens=30, reasoning_tokens=10)
    assert [config.continuation_token for config in configs] == [None, b"resume"]
    assert all(config.automatic_continuation is False for config in configs)


def test_gemini_answer_still_unfinished_after_the_last_hop_is_unparseable_with_its_tokens():
    configs: list[Any] = []
    with pytest.raises(LLMCallError) as caught:
        _generate(_gemini_hops([_hop([Part(text='{"move"')], FinishReason.CONTINUATION, b"resume")], configs), "gemini-3.6-flash", LLMCallOptions(thinking_enabled=True))

    assert len(configs) == _MAX_HOPS
    assert caught.value.kind is FailureKind.UNPARSEABLE
    assert caught.value.response is not None and caught.value.response.tokens.input_tokens == 100 * _MAX_HOPS


def test_gemini_block_on_a_later_hop_is_a_refusal_with_every_hop_paid_for():
    hops = [_hop([Part(text='{"mo')], FinishReason.CONTINUATION, b"resume"), _hop([], FinishReason.SAFETY)]
    with pytest.raises(LLMCallError) as caught:
        _generate(_gemini_hops(hops, []), "gemini-3.6-flash", LLMCallOptions(thinking_enabled=True))

    assert caught.value.kind is FailureKind.REFUSAL
    assert caught.value.response is not None and caught.value.response.tokens.input_tokens == 200


def test_gemini_hops_through_the_sdk_do_not_crash_on_a_thinking_only_first_hop():
    """The SDK's own hop following re-reads the joined answer against the schema and raised AttributeError
    (__pydantic_generic_metadata__, google-genai 2.31) when the first hop was thinking only. This runs the
    real SDK on fake HTTP."""
    token = base64.b64encode(b"resume").decode()
    bodies: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read().decode())
        if token not in bodies[-1]:
            candidate = {"finishReason": "CONTINUATION", "continuationToken": token, "content": {"role": "model", "parts": [{"text": "thinking", "thought": True}]}}
        else:
            candidate = {"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": '{"move": 4}'}]}}
        return httpx.Response(200, json={"candidates": [candidate], "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10}})

    sdk = genai.Client(api_key="fake", http_options=HttpOptions(httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))).aio
    response = _generate(GeminiLLMClient(sdk, concurrency=1), "gemini-3.6-flash", LLMCallOptions(thinking_enabled=True))

    assert len(bodies) == 2
    assert response.resolve_structured_output(_Move) == _Move(move=4)
    assert response.tokens.input_tokens == 200


def test_a_built_claude_client_sends_long_requests_without_streaming(monkeypatch):
    """The SDK refuses a non-streamed request whose max_tokens suggests over ten minutes, unless the client
    has its own timeout. `build()` always sets one; this keeps it that way."""
    recorder = _Recorder(_anthropic_body([{"type": "text", "text": '{"move": 4}'}]))
    real = AsyncAnthropic

    def with_mock_transport(**kwargs):
        return real(**kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)))

    monkeypatch.setattr("plyllm.providers.claude.client.AsyncAnthropic", with_mock_transport)
    client = ClaudeLLMClient.build(LLMConfig(claude=ClaudeProviderConfig(api_key="test")))
    assert client is not None

    # Opus 5.5 defaults to max_tokens=32000, well over what the SDK allows unstreamed by default
    _generate(client, "claude-opus-5.5", LLMCallOptions(thinking_enabled=True, reasoning_effort="max"))

    assert recorder.requests[0]["max_tokens"] == 32000
