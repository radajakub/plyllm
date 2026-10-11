"""OpenAI's Batch API behind LLM.batch: a JSONL file of /v1/responses requests in, an output file and an error
file back. The server is faked at the HTTP layer, so the real SDK builds every request and parses every answer."""

from __future__ import annotations

import asyncio
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

import plyllm.providers.openai.client as openai_client
from plyllm import LLM, BatchRequest, FailureKind, JsonBatchStore, LLMCallError, LLMCallOptions, LLMConfig, LLMMessage, LLMResponse, ModelConfig, Provider
from plyllm.batch import BatchJob
from plyllm.providers.openai.client import OpenAILLMClient


class _Move(BaseModel):
    move: int


_MODEL = ModelConfig(Provider.OPENAI, "gpt-5.4", LLMCallOptions(thinking_enabled=True, reasoning_effort="high"))


def _response_body(text: str, refusal: str | None = None) -> dict[str, Any]:
    content = [{"type": "refusal", "refusal": refusal}] if refusal is not None else [{"type": "output_text", "text": text, "annotations": []}]
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5.4-2026-03-05",
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "the centre wins"}]},
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": content},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "input_tokens_details": {"cached_tokens": 4}, "output_tokens_details": {"reasoning_tokens": 3}},
    }


def _output_line(custom_id: str, status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {"id": f"line_{custom_id}", "custom_id": custom_id, "response": {"status_code": status, "request_id": "req_1", "body": body}, "error": None}


def _error_line(custom_id: str, code: str, message: str) -> dict[str, Any]:
    return {"id": f"line_{custom_id}", "custom_id": custom_id, "response": None, "error": {"code": code, "message": message}}


class _FakeOpenAI:
    # the endpoints a batch touches, plus /v1/responses for comparing with generate()
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.batches: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.creates: list[dict[str, Any]] = []
        self.responses_requests: list[dict[str, Any]] = []
        self.response_body = _response_body('{"move": 4}')
        self.fail_next: dict[str, int] = {}  # path -> status to answer once

    def finish(self, batch_id: str, status: str = "completed", output: list[dict[str, Any]] | None = None, errors: list[dict[str, Any]] | None = None) -> None:
        batch = self.batches[batch_id]
        batch["status"] = status
        for key, lines in (("output_file_id", output), ("error_file_id", errors)):
            if lines is not None:
                file_id = f"file-{key}-{batch_id}"
                self.files[file_id] = "".join(json.dumps(line) + "\n" for line in lines).encode()
                batch[key] = file_id

    def uploaded_lines(self, batch_id: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.files[self.batches[batch_id]["input_file_id"]].decode().splitlines()]

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        method, path = request.method, request.url.path
        self.calls.append((method, path))
        if path in self.fail_next:
            return httpx2.Response(self.fail_next.pop(path), json={"error": {"message": "server error"}})

        if method == "POST" and path == "/v1/responses":
            self.responses_requests.append(json.loads(request.content))
            return httpx2.Response(200, json=self.response_body)
        if method == "POST" and path == "/v1/files":
            file_id = f"file-in-{len(self.files)}"
            self.files[file_id] = _uploaded_file(request)
            return httpx2.Response(
                200,
                json={"id": file_id, "object": "file", "bytes": len(self.files[file_id]), "created_at": 0, "filename": "batch.jsonl", "purpose": "batch", "status": "processed"},
            )
        if method == "POST" and path == "/v1/batches":
            body = json.loads(request.content)
            self.creates.append(body)
            batch_id = f"batch_{len(self.batches)}"
            self.batches[batch_id] = {
                "id": batch_id,
                "object": "batch",
                "endpoint": body["endpoint"],
                "completion_window": body["completion_window"],
                "input_file_id": body["input_file_id"],
                "created_at": 0,
                "status": "validating",
            }
            return httpx2.Response(200, json=self.batches[batch_id])
        if method == "GET" and path.startswith("/v1/batches/"):
            return httpx2.Response(200, json=self.batches[path.removeprefix("/v1/batches/")])
        if method == "POST" and path.endswith("/cancel"):
            batch = self.batches[path.removeprefix("/v1/batches/").removesuffix("/cancel")]
            batch["status"] = "cancelling"
            return httpx2.Response(200, json=batch)
        if method == "GET" and path.startswith("/v1/files/") and path.endswith("/content"):
            return httpx2.Response(200, content=self.files[path.removeprefix("/v1/files/").removesuffix("/content")])
        return httpx2.Response(404, json={"error": {"message": f"no fake for {method} {path}"}})


def _uploaded_file(request: httpx2.Request) -> bytes:
    message = BytesParser(policy=policy.default).parsebytes(b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.content)
    [part] = [part for part in message.iter_parts() if part.get_filename()]
    payload = part.get_payload(decode=True)
    assert isinstance(payload, bytes)
    return payload


@pytest.fixture
def server() -> _FakeOpenAI:
    return _FakeOpenAI()


@pytest.fixture
def llm(server: _FakeOpenAI) -> LLM:
    # max_retries=0: the SDK's own retries would sleep; plyllm's are what is under test
    sdk = AsyncOpenAI(api_key="test", max_retries=0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(server)))
    llm = LLM(LLMConfig())
    llm._provider_map[Provider.OPENAI] = OpenAILLMClient(sdk, concurrency=1)
    return llm


@pytest.fixture
def store(tmp_path: Path) -> JsonBatchStore:
    return JsonBatchStore(tmp_path / "batches.json")


def _requests(*ids: str) -> list[BatchRequest]:
    return [BatchRequest(request_id, LLMMessage.system("rules"), [LLMMessage.user(f"your move {request_id}")]) for request_id in ids]


async def _run(llm: LLM, store: JsonBatchStore, requests: list[BatchRequest], output_schema: type[BaseModel] | None = _Move):
    async with llm.batch(_MODEL, requests, store, output_schema=output_schema) as outcome:
        return outcome


# --- the same request and the same reading as generate() --------------------------------------------------


def test_a_batch_line_carries_exactly_the_body_generate_sends(llm: LLM, server: _FakeOpenAI, store: JsonBatchStore):
    [request] = _requests("q1")
    asyncio.run(llm.generate(_MODEL, request.system, request.messages, output_schema=_Move))

    asyncio.run(_run(llm, store, [request]))

    [line] = server.uploaded_lines("batch_0")
    assert line == {"custom_id": "q1", "method": "POST", "url": "/v1/responses", "body": server.responses_requests[0]}
    assert server.creates == [{"completion_window": "24h", "endpoint": "/v1/responses", "input_file_id": "file-in-0"}]
    assert store.load()[0].batch_id == "batch_0"


def test_a_collected_answer_reads_the_same_as_generate(llm: LLM, server: _FakeOpenAI, store: JsonBatchStore):
    [request] = _requests("q1")
    direct = asyncio.run(llm.generate(_MODEL, request.system, request.messages, output_schema=_Move))
    asyncio.run(_run(llm, store, [request]))
    server.finish("batch_0", output=[_output_line("q1", 200, server.response_body)])

    outcome = asyncio.run(_run(llm, store, [request]))

    collected = outcome.results["q1"]
    assert isinstance(collected, LLMResponse) and collected.batch and not direct.batch
    assert collected == LLMResponse(direct.provider, direct.model_string, direct.tokens, direct.items, direct.output_text, direct.structured_output_type, batch=True)
    assert store.load() == []


@pytest.mark.parametrize(
    ("status", "ended"),
    [("validating", False), ("in_progress", False), ("finalizing", False), ("cancelling", False), ("completed", True), ("expired", True), ("cancelled", True), ("failed", True)],
)
def test_a_batch_has_ended_once_the_provider_is_done_with_it(llm: LLM, server: _FakeOpenAI, store: JsonBatchStore, status: str, ended: bool):
    asyncio.run(_run(llm, store, _requests("q1")))
    server.batches["batch_0"]["status"] = status

    outcome = asyncio.run(_run(llm, store, _requests("q1")))

    assert (outcome.pending == set()) is ended


# --- what each request ends as --------------------------------------------------------------------------


def test_each_line_keeps_its_own_outcome(llm: LLM, server: _FakeOpenAI, store: JsonBatchStore):
    ids = ("ok", "refused", "throttled", "rejected", "expired", "unlisted")
    asyncio.run(_run(llm, store, _requests(*ids)))
    server.finish(
        "batch_0",
        status="expired",
        output=[
            _output_line("ok", 200, _response_body('{"move": 4}')),
            _output_line("refused", 200, _response_body("", refusal="I can't help with that.")),
            _output_line("throttled", 429, {"error": {"message": "Rate limit reached"}}),
            _output_line("rejected", 400, {"error": {"message": "Invalid schema"}}),
        ],
        errors=[_error_line("expired", "batch_expired", "This request could not be executed before the completion window expired.")],
    )

    outcome = asyncio.run(_run(llm, store, _requests(*ids)))

    assert isinstance(outcome.results["ok"], LLMResponse) and outcome.results["ok"].resolve_structured_output(_Move) == _Move(move=4)
    kinds = {request_id: result.kind for request_id, result in outcome.results.items() if isinstance(result, LLMCallError)}
    # "unlisted" is in neither file; the batch expired, so it never ran
    assert kinds == {
        "refused": FailureKind.REFUSAL,
        "throttled": FailureKind.RATE_LIMIT,
        "rejected": FailureKind.PROVIDER,
        "expired": FailureKind.EXPIRED,
        "unlisted": FailureKind.EXPIRED,
    }
    refused = outcome.results["refused"]
    assert isinstance(refused, LLMCallError) and refused.response is not None and refused.response.batch


def test_a_cancelled_batch_ends_its_unrun_requests_as_cancelled(llm: LLM, server: _FakeOpenAI, store: JsonBatchStore):
    asyncio.run(_run(llm, store, _requests("done", "never")))
    asyncio.run(llm.cancel_batches(Provider.OPENAI, store))
    assert ("POST", "/v1/batches/batch_0/cancel") in server.calls
    server.finish("batch_0", status="cancelled", output=[_output_line("done", 200, _response_body('{"move": 1}'))])

    outcome = asyncio.run(_run(llm, store, _requests("done", "never")))

    never = outcome.results["never"]
    assert isinstance(outcome.results["done"], LLMResponse)
    assert isinstance(never, LLMCallError) and never.kind is FailureKind.CANCELLED


def test_a_batch_that_failed_validation_fails_every_request_with_the_reason(llm: LLM, server: _FakeOpenAI, store: JsonBatchStore):
    asyncio.run(_run(llm, store, _requests("q1", "q2")))
    server.finish("batch_0", status="failed")
    server.batches["batch_0"]["errors"] = {"object": "list", "data": [{"code": "invalid_request", "line": 1, "message": "Model not available in batch"}]}

    outcome = asyncio.run(_run(llm, store, _requests("q1", "q2")))

    for result in outcome.results.values():
        assert isinstance(result, LLMCallError) and result.kind is FailureKind.PROVIDER and "Model not available in batch" in str(result)


# --- submitting -----------------------------------------------------------------------------------------------


def test_the_upload_is_retried_but_the_create_is_not(llm: LLM, server: _FakeOpenAI, store: JsonBatchStore, monkeypatch: pytest.MonkeyPatch):
    async def no_wait(_delay: float) -> None:
        return None

    monkeypatch.setattr("plyllm.concurrency.asyncio.sleep", no_wait)
    server.fail_next = {"/v1/files": 500}

    asyncio.run(_run(llm, store, _requests("q1")))
    assert server.calls.count(("POST", "/v1/files")) == 2 and len(server.batches) == 1

    server.fail_next = {"/v1/batches": 500}
    with pytest.raises(LLMCallError) as caught:
        asyncio.run(_run(llm, store, _requests("q2")))

    # a 500 is an answer: no batch was made, so the record goes; a timeout would keep it (tests/test_batch.py)
    assert caught.value.kind is FailureKind.PROVIDER
    assert server.calls.count(("POST", "/v1/batches")) == 2
    assert [list(job.requests) for job in store.load()] == [["q1"]]


def test_an_input_over_the_size_limit_is_split_into_files_that_fit(llm: LLM, server: _FakeOpenAI, store: JsonBatchStore, monkeypatch: pytest.MonkeyPatch):
    one_line = len(openai_client._input_line("q1", llm._provider_map[Provider.OPENAI]._request(*_request_args("q1"))))
    monkeypatch.setattr(OpenAILLMClient, "max_batch_bytes", 2 * one_line + 10)

    asyncio.run(_run(llm, store, _requests("q1", "q2", "q3")))

    assert [[line["custom_id"] for line in server.uploaded_lines(batch_id)] for batch_id in server.batches] == [["q1", "q2"], ["q3"]]
    assert all(len(content) <= 2 * one_line + 10 for content in server.files.values())


def _request_args(request_id: str) -> tuple[Any, ...]:
    [request] = _requests(request_id)
    return (next(m for m in openai_client.openai_models() if m.model_name == _MODEL.model_name), request.system, request.messages, _MODEL.options, _Move)


def test_the_store_records_the_openai_job(llm: LLM, store: JsonBatchStore):
    asyncio.run(_run(llm, store, _requests("q1", "q2")))

    [job] = store.load()
    assert isinstance(job, BatchJob) and job.provider is Provider.OPENAI and job.model_name == "gpt-5.4" and list(job.requests) == ["q1", "q2"]
