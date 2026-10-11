"""Anthropic's Message Batches behind LLM.batch: the requests go inline with the create call, the results come back as
a JSONL stream. The server is faked at the HTTP layer, so the real SDK builds every request and parses every answer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from plyllm import LLM, BatchRequest, FailureKind, JsonBatchStore, LLMCallError, LLMCallOptions, LLMConfig, LLMMessage, LLMResponse, ModelConfig, Provider
from plyllm.providers.claude.client import ClaudeLLMClient


class _Move(BaseModel):
    move: int


_MODEL = ModelConfig(Provider.CLAUDE, "claude-opus-5", LLMCallOptions(thinking_enabled=True, reasoning_effort="high"))


def _message(text: str, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "content": [{"type": "thinking", "thinking": "the centre wins", "signature": "sig"}, {"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 0, "output_tokens_details": {"thinking_tokens": 3}},
    }


def _succeeded(custom_id: str, message: dict[str, Any]) -> dict[str, Any]:
    return {"custom_id": custom_id, "result": {"type": "succeeded", "message": message}}


def _errored(custom_id: str, error_type: str, message: str) -> dict[str, Any]:
    return {"custom_id": custom_id, "result": {"type": "errored", "error": {"type": "error", "error": {"type": error_type, "message": message}}}}


class _FakeAnthropic:
    # the endpoints a batch touches, plus /v1/messages for comparing with generate()
    def __init__(self) -> None:
        self.batches: dict[str, dict[str, Any]] = {}
        self.results: dict[str, list[dict[str, Any]]] = {}
        self.creates: list[dict[str, Any]] = []
        self.messages_requests: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str]] = []
        self.message = _message('{"move": 4}')

    def finish(self, batch_id: str, lines: list[dict[str, Any]]) -> None:
        self.batches[batch_id].update(processing_status="ended", ended_at="2026-10-11T11:00:00Z", results_url=f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results")
        self.results[batch_id] = lines

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        method, path = request.method, request.url.path
        self.calls.append((method, path))
        if method == "POST" and path == "/v1/messages":
            self.messages_requests.append(json.loads(request.content))
            return httpx2.Response(200, json=self.message)
        if method == "POST" and path == "/v1/messages/batches":
            self.creates.append(json.loads(request.content))
            batch_id = f"msgbatch_{len(self.batches)}"
            self.batches[batch_id] = {
                "id": batch_id,
                "type": "message_batch",
                "processing_status": "in_progress",
                "request_counts": {"processing": len(self.creates[-1]["requests"]), "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0},
                "created_at": "2026-10-11T10:00:00Z",
                "expires_at": "2026-10-12T10:00:00Z",
                "ended_at": None,
                "archived_at": None,
                "cancel_initiated_at": None,
                "results_url": None,
            }
            return httpx2.Response(200, json=self.batches[batch_id])
        if method == "GET" and path.endswith("/results"):
            batch_id = path.removeprefix("/v1/messages/batches/").removesuffix("/results")
            return httpx2.Response(200, content="".join(json.dumps(line) + "\n" for line in self.results[batch_id]).encode())
        if method == "GET" and path.startswith("/v1/messages/batches/"):
            return httpx2.Response(200, json=self.batches[path.removeprefix("/v1/messages/batches/")])
        if method == "POST" and path.endswith("/cancel"):
            batch = self.batches[path.removeprefix("/v1/messages/batches/").removesuffix("/cancel")]
            batch["processing_status"] = "canceling"
            return httpx2.Response(200, json=batch)
        return httpx2.Response(404, json={"type": "error", "error": {"type": "not_found_error", "message": f"no fake for {method} {path}"}})


@pytest.fixture
def server() -> _FakeAnthropic:
    return _FakeAnthropic()


@pytest.fixture
def llm(server: _FakeAnthropic) -> LLM:
    # max_retries=0: the SDK's own retries would sleep; plyllm's are what is under test
    sdk = AsyncAnthropic(api_key="test", max_retries=0, timeout=60.0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(server)))
    llm = LLM(LLMConfig())
    llm._provider_map[Provider.CLAUDE] = ClaudeLLMClient(sdk, concurrency=1)
    return llm


@pytest.fixture
def store(tmp_path: Path) -> JsonBatchStore:
    return JsonBatchStore(tmp_path / "batches.json")


def _requests(*ids: str) -> list[BatchRequest]:
    return [BatchRequest(request_id, LLMMessage.system("rules"), [LLMMessage.user(f"your move {request_id}")]) for request_id in ids]


async def _run(llm: LLM, store: JsonBatchStore, requests: list[BatchRequest]):
    async with llm.batch(_MODEL, requests, store, output_schema=_Move) as outcome:
        return outcome


# --- the same request and the same reading as generate() --------------------------------------------------


def test_a_batch_request_carries_exactly_the_params_generate_sends(llm: LLM, server: _FakeAnthropic, store: JsonBatchStore):
    [request] = _requests("q1")
    asyncio.run(llm.generate(_MODEL, request.system, request.messages, output_schema=_Move))

    asyncio.run(_run(llm, store, [request]))

    assert server.creates == [{"requests": [{"custom_id": "q1", "params": server.messages_requests[0]}]}]
    assert store.load()[0].batch_id == "msgbatch_0"


def test_a_collected_answer_reads_the_same_as_generate(llm: LLM, server: _FakeAnthropic, store: JsonBatchStore):
    [request] = _requests("q1")
    direct = asyncio.run(llm.generate(_MODEL, request.system, request.messages, output_schema=_Move))
    asyncio.run(_run(llm, store, [request]))
    server.finish("msgbatch_0", [_succeeded("q1", server.message)])

    outcome = asyncio.run(_run(llm, store, [request]))

    collected = outcome.results["q1"]
    assert isinstance(collected, LLMResponse)
    assert collected == LLMResponse(direct.provider, direct.model_string, direct.tokens, direct.items, direct.output_text, direct.structured_output_type, batch=True)
    assert store.load() == []


@pytest.mark.parametrize(("status", "ended"), [("in_progress", False), ("canceling", False), ("ended", True)])
def test_a_batch_has_ended_once_its_processing_has(llm: LLM, server: _FakeAnthropic, store: JsonBatchStore, status: str, ended: bool):
    asyncio.run(_run(llm, store, _requests("q1")))
    server.finish("msgbatch_0", [_succeeded("q1", server.message)])
    server.batches["msgbatch_0"]["processing_status"] = status

    outcome = asyncio.run(_run(llm, store, _requests("q1")))

    assert (outcome.pending == set()) is ended


# --- what each request ends as --------------------------------------------------------------------------


def test_each_result_keeps_its_own_outcome(llm: LLM, server: _FakeAnthropic, store: JsonBatchStore):
    ids = ("ok", "refused", "invalid", "throttled", "late", "stopped")
    asyncio.run(_run(llm, store, _requests(*ids)))
    server.finish(
        "msgbatch_0",
        [
            _succeeded("ok", _message('{"move": 4}')),
            _succeeded("refused", _message("I won't do that.", stop_reason="refusal")),
            _errored("invalid", "invalid_request_error", "max_tokens: too large"),
            _errored("throttled", "rate_limit_error", "Rate limited"),
            {"custom_id": "late", "result": {"type": "expired"}},
            {"custom_id": "stopped", "result": {"type": "canceled"}},
        ],
    )

    outcome = asyncio.run(_run(llm, store, _requests(*ids)))

    ok = outcome.results["ok"]
    assert isinstance(ok, LLMResponse) and ok.resolve_structured_output(_Move) == _Move(move=4)
    kinds = {request_id: result.kind for request_id, result in outcome.results.items() if isinstance(result, LLMCallError)}
    assert kinds == {
        "refused": FailureKind.REFUSAL,
        "invalid": FailureKind.PROVIDER,
        "throttled": FailureKind.RATE_LIMIT,
        "late": FailureKind.EXPIRED,
        "stopped": FailureKind.CANCELLED,
    }
    assert "max_tokens: too large" in str(outcome.results["invalid"])


def test_cancel_asks_anthropic_to_stop_the_batch(llm: LLM, server: _FakeAnthropic, store: JsonBatchStore):
    asyncio.run(_run(llm, store, _requests("q1")))

    asyncio.run(llm.cancel_batches(Provider.CLAUDE, store))

    assert ("POST", "/v1/messages/batches/msgbatch_0/cancel") in server.calls


# --- submitting -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("request_id", ["has space", "x" * 65, "", "dots.are.out"])
def test_an_id_anthropic_would_reject_is_refused_before_anything_is_sent(llm: LLM, server: _FakeAnthropic, store: JsonBatchStore, request_id: str):
    with pytest.raises(ValueError, match="1-64 letters"):
        llm.batch(_MODEL, _requests(request_id), store)

    assert server.calls == []


def test_requests_over_the_size_limit_are_split_into_batches_that_fit(llm: LLM, server: _FakeAnthropic, store: JsonBatchStore, monkeypatch: pytest.MonkeyPatch):
    client = llm._provider_map[Provider.CLAUDE]
    [request] = _requests("q1")
    model = client.resolve_model(_MODEL.model_name)
    one = client._batch_size("q1", client._request(model, request.system, request.messages, _MODEL.options, _Move))
    monkeypatch.setattr(ClaudeLLMClient, "max_batch_bytes", 2 * one + 10)

    asyncio.run(_run(llm, store, _requests("q1", "q2", "q3")))

    assert [[item["custom_id"] for item in create["requests"]] for create in server.creates] == [["q1", "q2"], ["q3"]]
