"""Gemini's batch mode behind LLM.batch: inline requests in, inlined responses back, matched by the id each request
carries in its metadata. The server is faked at the HTTP layer, so the real SDK builds every request and parses every
answer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from google import genai
from google.genai.types import HttpOptions
from pydantic import BaseModel

from plyllm import LLM, BatchRequest, FailureKind, JsonBatchStore, LLMCallError, LLMCallOptions, LLMConfig, LLMMessage, LLMResponse, ModelConfig, Provider
from plyllm.providers.gemini.client import GeminiLLMClient


class _Move(BaseModel):
    move: int


_MODEL = ModelConfig(Provider.GEMINI, "gemini-3.5-flash", LLMCallOptions(thinking_enabled=True))


def _response(text: str, finish_reason: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "the centre wins", "thought": True}, {"text": text}]}, "finishReason": finish_reason}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 3, "cachedContentTokenCount": 4},
    }


class _FakeGemini:
    # the endpoints a batch touches, plus generateContent for comparing with generate()
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.creates: list[dict[str, Any]] = []
        self.generate_requests: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str]] = []
        self.answer = _response('{"move": 4}')

    def finish(self, name: str, responses: list[dict[str, Any]] | None, state: str = "BATCH_STATE_SUCCEEDED", error: str | None = None) -> None:
        metadata = self.jobs[name]["metadata"]
        metadata["state"] = state
        if responses is not None:
            metadata["output"] = {"inlinedResponses": {"inlinedResponses": responses}}
        if error is not None:
            self.jobs[name]["error"] = {"code": 3, "message": error}

    def sent_requests(self, index: int) -> list[dict[str, Any]]:
        return self.creates[index]["batch"]["inputConfig"]["requests"]["requests"]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.calls.append((method, path))
        if path.endswith(":generateContent"):
            self.generate_requests.append(json.loads(request.content))
            return httpx.Response(200, json=self.answer)
        if path.endswith(":batchGenerateContent"):
            self.creates.append(json.loads(request.content))
            name = f"batches/b{len(self.jobs)}"
            self.jobs[name] = {"name": name, "metadata": {"state": "BATCH_STATE_PENDING", "model": path.removeprefix("/v1beta/").removesuffix(":batchGenerateContent")}}
            return httpx.Response(200, json=self.jobs[name])
        if method == "POST" and path.endswith(":cancel"):
            self.jobs[path.removeprefix("/v1beta/").removesuffix(":cancel")]["metadata"]["state"] = "BATCH_STATE_CANCELLED"
            return httpx.Response(200, json={})
        if method == "GET" and path.startswith("/v1beta/batches/"):
            return httpx.Response(200, json=self.jobs[path.removeprefix("/v1beta/")])
        return httpx.Response(404, json={"error": {"code": 404, "message": f"no fake for {method} {path}", "status": "NOT_FOUND"}})


@pytest.fixture
def server() -> _FakeGemini:
    return _FakeGemini()


@pytest.fixture
def llm(server: _FakeGemini) -> LLM:
    http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    sdk = genai.Client(api_key="test", http_options=HttpOptions(httpx_async_client=http)).aio
    llm = LLM(LLMConfig())
    llm._provider_map[Provider.GEMINI] = GeminiLLMClient(sdk, concurrency=1)
    return llm


@pytest.fixture
def store(tmp_path: Path) -> JsonBatchStore:
    return JsonBatchStore(tmp_path / "batches.json")


def _requests(*ids: str) -> list[BatchRequest]:
    return [BatchRequest(request_id, LLMMessage.system("rules"), [LLMMessage.user(f"your move {request_id}")]) for request_id in ids]


async def _run(llm: LLM, store: JsonBatchStore, requests: list[BatchRequest]):
    async with llm.batch(_MODEL, requests, store, output_schema=_Move) as outcome:
        return outcome


def _inlined(request_id: str | None, response: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {}
    if request_id is not None:
        item["metadata"] = {"plyllm_id": request_id}
    if response is not None:
        item["response"] = response
    if error is not None:
        item["error"] = {"code": 3, "message": error}
    return item


# --- the same request and the same reading as generate() --------------------------------------------------


def test_an_inline_request_carries_exactly_the_request_generate_sends(llm: LLM, server: _FakeGemini, store: JsonBatchStore):
    [request] = _requests("q1")
    asyncio.run(llm.generate(_MODEL, request.system, request.messages, output_schema=_Move))

    asyncio.run(_run(llm, store, [request]))

    assert server.sent_requests(0) == [{"request": server.generate_requests[0], "metadata": {"plyllm_id": "q1"}}]
    assert ("POST", "/v1beta/models/gemini-3.5-flash:batchGenerateContent") in server.calls
    assert store.load()[0].batch_id == "batches/b0"


def test_a_collected_answer_reads_the_same_as_generate(llm: LLM, server: _FakeGemini, store: JsonBatchStore):
    [request] = _requests("q1")
    direct = asyncio.run(llm.generate(_MODEL, request.system, request.messages, output_schema=_Move))
    asyncio.run(_run(llm, store, [request]))
    server.finish("batches/b0", [_inlined("q1", server.answer)])

    outcome = asyncio.run(_run(llm, store, [request]))

    collected = outcome.results["q1"]
    assert isinstance(collected, LLMResponse)
    assert collected == LLMResponse(direct.provider, direct.model_string, direct.tokens, direct.items, direct.output_text, direct.structured_output_type, batch=True)
    assert store.load() == []


@pytest.mark.parametrize(
    ("state", "ended"),
    [
        ("BATCH_STATE_PENDING", False),
        ("BATCH_STATE_RUNNING", False),
        ("BATCH_STATE_SUCCEEDED", True),
        ("BATCH_STATE_FAILED", True),
        ("BATCH_STATE_CANCELLED", True),
        ("BATCH_STATE_EXPIRED", True),
    ],
)
def test_a_batch_has_ended_once_its_job_is_done(llm: LLM, server: _FakeGemini, store: JsonBatchStore, state: str, ended: bool):
    asyncio.run(_run(llm, store, _requests("q1")))
    server.jobs["batches/b0"]["metadata"]["state"] = state

    outcome = asyncio.run(_run(llm, store, _requests("q1")))

    assert (outcome.pending == set()) is ended


# --- what each request ends as --------------------------------------------------------------------------


def test_each_response_keeps_its_own_outcome(llm: LLM, server: _FakeGemini, store: JsonBatchStore):
    asyncio.run(_run(llm, store, _requests("ok", "blocked", "failed", "unlabelled")))
    server.finish(
        "batches/b0",
        [
            _inlined("ok", _response('{"move": 4}')),
            _inlined("blocked", _response("", finish_reason="SAFETY")),
            _inlined("failed", error="Request contains an invalid argument."),
            # no echoed id: matched by position, the fourth request
            _inlined(None, _response('{"move": 7}')),
        ],
    )

    outcome = asyncio.run(_run(llm, store, _requests("ok", "blocked", "failed", "unlabelled")))

    kinds = {request_id: result.kind for request_id, result in outcome.results.items() if isinstance(result, LLMCallError)}
    assert kinds == {"blocked": FailureKind.REFUSAL, "failed": FailureKind.PROVIDER}
    unlabelled = outcome.results["unlabelled"]
    assert isinstance(unlabelled, LLMResponse) and unlabelled.resolve_structured_output(_Move) == _Move(move=7)


@pytest.mark.parametrize(
    ("state", "kind"), [("BATCH_STATE_EXPIRED", FailureKind.EXPIRED), ("BATCH_STATE_CANCELLED", FailureKind.CANCELLED), ("BATCH_STATE_FAILED", FailureKind.PROVIDER)]
)
def test_a_job_that_ended_without_answers_fails_every_request_by_its_state(llm: LLM, server: _FakeGemini, store: JsonBatchStore, state: str, kind: FailureKind):
    asyncio.run(_run(llm, store, _requests("q1", "q2")))
    server.finish("batches/b0", None, state=state, error="quota exceeded" if state == "BATCH_STATE_FAILED" else None)

    outcome = asyncio.run(_run(llm, store, _requests("q1", "q2")))

    for result in outcome.results.values():
        assert isinstance(result, LLMCallError) and result.kind is kind


def test_a_cancelled_job_keeps_the_answers_it_finished(llm: LLM, server: _FakeGemini, store: JsonBatchStore):
    asyncio.run(_run(llm, store, _requests("done", "never")))
    asyncio.run(llm.cancel_batches(Provider.GEMINI, store))
    assert ("POST", "/v1beta/batches/b0:cancel") in server.calls
    server.finish("batches/b0", [_inlined("done", _response('{"move": 1}'))], state="BATCH_STATE_CANCELLED")

    outcome = asyncio.run(_run(llm, store, _requests("done", "never")))

    never = outcome.results["never"]
    assert isinstance(outcome.results["done"], LLMResponse)
    assert isinstance(never, LLMCallError) and never.kind is FailureKind.CANCELLED


# --- submitting -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("system", "user"), [("rules", "go"), ("rules " * 2000, "your move " * 3000)])
def test_the_size_estimate_does_not_undercount_what_the_sdk_sends(llm: LLM, server: _FakeGemini, store: JsonBatchStore, system: str, user: str):
    # the margin under the 20 MB limit holds only if no request is counted smaller than the SDK sends it
    [request] = [BatchRequest("q1", LLMMessage.system(system), [LLMMessage.user(user)])]
    client = llm._provider_map[Provider.GEMINI]
    model = client.resolve_model(_MODEL.model_name)
    estimate = client._batch_size("q1", client._request(model, request.system, request.messages, _MODEL.options, _Move))

    asyncio.run(_run(llm, store, [request]))

    sent = len(json.dumps(server.sent_requests(0)[0], ensure_ascii=False).encode())
    assert sent <= estimate <= 1.5 * sent


def test_requests_over_the_size_limit_are_split_into_batches_that_fit(llm: LLM, server: _FakeGemini, store: JsonBatchStore, monkeypatch: pytest.MonkeyPatch):
    client = llm._provider_map[Provider.GEMINI]
    [request] = _requests("q1")
    model = client.resolve_model(_MODEL.model_name)
    one = client._batch_size("q1", client._request(model, request.system, request.messages, _MODEL.options, _Move))
    monkeypatch.setattr(GeminiLLMClient, "max_batch_bytes", 2 * one + 10)

    asyncio.run(_run(llm, store, _requests("q1", "q2", "q3")))

    assert [[item["metadata"]["plyllm_id"] for item in server.sent_requests(index)] for index in range(len(server.creates))] == [["q1", "q2"], ["q3"]]
