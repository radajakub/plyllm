"""Mistral's batch jobs behind LLM.batch: inline requests in, an output file and an error file back. The server is faked
at the HTTP layer, so the real SDK builds every request and parses every answer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from mistralai.client import Mistral
from pydantic import BaseModel

from plyllm import LLM, BatchRequest, FailureKind, JsonBatchStore, LLMCallError, LLMCallOptions, LLMConfig, LLMMessage, LLMResponse, ModelConfig, Provider
from plyllm.providers.mistral.client import MistralLLMClient


class _Move(BaseModel):
    move: int


_MODEL = ModelConfig(Provider.MISTRAL, "mistral-small-4", LLMCallOptions(thinking_enabled=True, reasoning_effort="high"))


def _completion(text: str) -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "model": "mistral-small-2603",
        "created": 0,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
    }


def _output_line(custom_id: str, status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {"id": f"line-{custom_id}", "custom_id": custom_id, "response": {"status_code": status, "body": body}, "error": None}


class _FakeMistral:
    # the endpoints a batch touches, plus chat completions for comparing with generate()
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.files: dict[str, bytes] = {}
        self.creates: list[dict[str, Any]] = []
        self.chat_requests: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str]] = []
        self.answer = _completion('{"move": 4}')

    def finish(self, job_id: str, status: str = "SUCCESS", output: list[dict[str, Any]] | None = None, errors: list[dict[str, Any]] | None = None) -> None:
        job = self.jobs[job_id]
        job["status"] = status
        for key, lines in (("output_file", output), ("error_file", errors)):
            if lines is not None:
                file_id = f"file-{key}-{job_id}"
                self.files[file_id] = "".join(json.dumps(line) + "\n" for line in lines).encode()
                job[key] = file_id

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        method, path = request.method, request.url.path
        self.calls.append((method, path))
        if method == "POST" and path == "/v1/chat/completions":
            self.chat_requests.append(json.loads(request.content))
            return httpx2.Response(200, json=self.answer)
        if method == "POST" and path == "/v1/batch/jobs":
            body = json.loads(request.content)
            self.creates.append(body)
            job_id = f"job-{len(self.jobs)}"
            self.jobs[job_id] = {
                "id": job_id,
                "object": "batch",
                "input_files": [],
                "endpoint": body["endpoint"],
                "model": body.get("model"),
                "errors": [],
                "status": "QUEUED",
                "created_at": 0,
                "total_requests": len(body["requests"]),
                "completed_requests": 0,
                "succeeded_requests": 0,
                "failed_requests": 0,
            }
            return httpx2.Response(200, json=self.jobs[job_id])
        if method == "POST" and path.endswith("/cancel"):
            job = self.jobs[path.removeprefix("/v1/batch/jobs/").removesuffix("/cancel")]
            job["status"] = "CANCELLATION_REQUESTED"
            return httpx2.Response(200, json=job)
        if method == "GET" and path.startswith("/v1/batch/jobs/"):
            return httpx2.Response(200, json=self.jobs[path.removeprefix("/v1/batch/jobs/")])
        if method == "GET" and path.startswith("/v1/files/") and path.endswith("/content"):
            return httpx2.Response(200, content=self.files[path.removeprefix("/v1/files/").removesuffix("/content")], headers={"content-type": "application/octet-stream"})
        return httpx2.Response(404, json={"message": f"no fake for {method} {path}"})


@pytest.fixture
def server() -> _FakeMistral:
    return _FakeMistral()


@pytest.fixture
def llm(server: _FakeMistral) -> LLM:
    sdk = Mistral(api_key="test", async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(server)))
    llm = LLM(LLMConfig())
    llm._provider_map[Provider.MISTRAL] = MistralLLMClient(sdk, concurrency=1)
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


def test_a_batch_request_carries_exactly_the_body_generate_sends(llm: LLM, server: _FakeMistral, store: JsonBatchStore):
    [request] = _requests("q1")
    asyncio.run(llm.generate(_MODEL, request.system, request.messages, output_schema=_Move))

    asyncio.run(_run(llm, store, [request]))

    [create] = server.creates
    assert create["requests"] == [{"custom_id": "q1", "body": server.chat_requests[0]}]
    assert create["endpoint"] == "/v1/chat/completions" and create["model"] == "mistral-small-2603"
    assert store.load()[0].batch_id == "job-0"


def test_a_collected_answer_reads_the_same_as_generate(llm: LLM, server: _FakeMistral, store: JsonBatchStore):
    [request] = _requests("q1")
    direct = asyncio.run(llm.generate(_MODEL, request.system, request.messages, output_schema=_Move))
    asyncio.run(_run(llm, store, [request]))
    server.finish("job-0", output=[_output_line("q1", 200, server.answer)])

    outcome = asyncio.run(_run(llm, store, [request]))

    collected = outcome.results["q1"]
    assert isinstance(collected, LLMResponse)
    assert collected == LLMResponse(direct.provider, direct.model_string, direct.tokens, direct.items, direct.output_text, direct.structured_output_type, batch=True)
    assert store.load() == []


@pytest.mark.parametrize(
    ("status", "ended"),
    [("QUEUED", False), ("RUNNING", False), ("CANCELLATION_REQUESTED", False), ("SUCCESS", True), ("FAILED", True), ("TIMEOUT_EXCEEDED", True), ("CANCELLED", True)],
)
def test_a_batch_has_ended_once_its_job_has(llm: LLM, server: _FakeMistral, store: JsonBatchStore, status: str, ended: bool):
    asyncio.run(_run(llm, store, _requests("q1")))
    server.jobs["job-0"]["status"] = status

    outcome = asyncio.run(_run(llm, store, _requests("q1")))

    assert (outcome.pending == set()) is ended


# --- what each request ends as --------------------------------------------------------------------------


def test_each_line_keeps_its_own_outcome(llm: LLM, server: _FakeMistral, store: JsonBatchStore):
    ids = ("ok", "throttled", "invalid", "failed", "unlisted")
    asyncio.run(_run(llm, store, _requests(*ids)))
    server.finish(
        "job-0",
        status="TIMEOUT_EXCEEDED",
        output=[
            _output_line("ok", 200, _completion('{"move": 4}')),
            _output_line("throttled", 429, {"message": "Requests rate limit exceeded"}),
            _output_line("invalid", 400, {"message": "Invalid response_format"}),
        ],
        errors=[{"id": "line-failed", "custom_id": "failed", "response": None, "error": {"message": "internal error"}}],
    )

    outcome = asyncio.run(_run(llm, store, _requests(*ids)))

    ok = outcome.results["ok"]
    assert isinstance(ok, LLMResponse) and ok.resolve_structured_output(_Move) == _Move(move=4)
    kinds = {request_id: result.kind for request_id, result in outcome.results.items() if isinstance(result, LLMCallError)}
    # "unlisted" is in neither file; the job ran out of time, so it never ran
    assert kinds == {"throttled": FailureKind.RATE_LIMIT, "invalid": FailureKind.PROVIDER, "failed": FailureKind.PROVIDER, "unlisted": FailureKind.EXPIRED}
    assert "Invalid response_format" in str(outcome.results["invalid"])


def test_a_cancelled_job_ends_its_unrun_requests_as_cancelled(llm: LLM, server: _FakeMistral, store: JsonBatchStore):
    asyncio.run(_run(llm, store, _requests("done", "never")))
    asyncio.run(llm.cancel_batches(Provider.MISTRAL, store))
    assert ("POST", "/v1/batch/jobs/job-0/cancel") in server.calls
    server.finish("job-0", status="CANCELLED", output=[_output_line("done", 200, _completion('{"move": 1}'))])

    outcome = asyncio.run(_run(llm, store, _requests("done", "never")))

    never = outcome.results["never"]
    assert isinstance(outcome.results["done"], LLMResponse)
    assert isinstance(never, LLMCallError) and never.kind is FailureKind.CANCELLED


def test_a_failed_job_fails_every_request_with_its_reasons(llm: LLM, server: _FakeMistral, store: JsonBatchStore):
    asyncio.run(_run(llm, store, _requests("q1", "q2")))
    server.finish("job-0", status="FAILED")
    server.jobs["job-0"]["errors"] = [{"message": "Model not found", "count": 2}]

    outcome = asyncio.run(_run(llm, store, _requests("q1", "q2")))

    for result in outcome.results.values():
        assert isinstance(result, LLMCallError) and result.kind is FailureKind.PROVIDER and "Model not found" in str(result)


# --- submitting -----------------------------------------------------------------------------------------------


def test_more_requests_than_one_inline_job_takes_are_split(llm: LLM, server: _FakeMistral, store: JsonBatchStore, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MistralLLMClient, "max_batch_requests", 2)

    asyncio.run(_run(llm, store, _requests("q1", "q2", "q3")))

    assert [[item["custom_id"] for item in create["requests"]] for create in server.creates] == [["q1", "q2"], ["q3"]]
