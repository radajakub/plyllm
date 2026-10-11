"""The provider-independent batch layer: one run submits what is new, collects what has ended and leaves the
rest pending; the store holds only the jobs still at the provider. The provider here is a fake that keeps its
batches in memory, so every test runs offline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from plyllm import LLM, BatchRequest, FailureKind, JsonBatchStore, LLMCallError, LLMCallOptions, LLMConfig, LLMMessage, LLMResponse, LLMTokens, ModelConfig, Provider
from plyllm.batch import DEFAULT_POLL_INTERVAL, BatchJob, request_hash
from plyllm.batch_run import BatchRun, cancel_batches
from plyllm.client import LLMClient
from plyllm.model import LLMModel


class _Move(BaseModel):
    move: int


class _StubModel(LLMModel):
    def extract_params(self, options: LLMCallOptions) -> dict:
        self.validate(options)
        return {}


class _StatusError(Exception):
    pass


class _FakeBatchClient(LLMClient[_StubModel, str]):
    """A provider whose batch API lives in memory. A request's answer is its prompt, unless `answers` says otherwise."""

    provider_key = Provider.OPENAI
    _retry_errors = (ConnectionError,)
    supports_batch = True
    max_batch_requests = 2
    max_batch_bytes = 1_000

    def __init__(self) -> None:
        models = [_StubModel("stub", "stub-v1", batch_ratio=0.5), _StubModel("no-batch-price", "no-batch-price-v1")]
        super().__init__(models, [], concurrency=1)
        self.batches: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self.ended: set[str] = set()
        self.answers: dict[str, str | LLMCallError | None] = {}
        self.submit_error: Exception | None = None
        self.status_checks = 0
        self.end_after_checks: int | None = None
        self.cancelled: list[str] = []

    @classmethod
    def build(cls, config: LLMConfig) -> _FakeBatchClient:
        return cls()

    def _should_retry_on_error(self, error: Exception) -> bool:
        return False

    def error_kind(self, error: Exception) -> FailureKind:
        if isinstance(error, TimeoutError):
            return FailureKind.TIMEOUT
        return FailureKind.PROVIDER if isinstance(error, _StatusError) else FailureKind.OTHER

    def _request(self, model: _StubModel, system: LLMMessage, messages: list[LLMMessage], options: LLMCallOptions, output_schema: type[BaseModel] | None) -> dict[str, Any]:
        model.extract_params(options)
        return {"system": system.content, "prompt": messages[-1].content}

    async def _send(self, kwargs: dict[str, Any]) -> str:
        return kwargs["prompt"]

    def _tokens_of(self, response: str) -> int:
        return 1

    def _parse(self, model: _StubModel, response: str, output_schema: type[BaseModel] | None) -> LLMResponse:
        return self._answer(model, response, [], LLMTokens(input_tokens=3, output_tokens=2), output_schema)

    def _batch_size(self, request_id: str, body: dict[str, Any]) -> int:
        return len(body["prompt"])

    async def _submit_batch(self, model: _StubModel, requests: list[tuple[str, dict[str, Any]]]) -> str:
        if self.submit_error is not None:
            raise self.submit_error
        batch_id = f"batch-{len(self.batches)}"
        self.batches[batch_id] = requests
        return batch_id

    async def _batch_ended(self, batch_id: str) -> bool:
        self.status_checks += 1
        if self.end_after_checks is not None and self.status_checks >= self.end_after_checks:
            self.ended.update(self.batches)
        return batch_id in self.ended

    async def _batch_results(self, job: BatchJob) -> dict[str, str | LLMCallError]:
        assert job.batch_id is not None
        results: dict[str, str | LLMCallError] = {}
        for request_id, body in self.batches[job.batch_id]:
            answer = self.answers.get(request_id, body["prompt"])
            if answer is not None:  # None: the provider lost the request
                results[request_id] = answer
        return results

    async def _cancel_batch(self, batch_id: str) -> None:
        self.cancelled.append(batch_id)

    def end_all(self) -> None:
        self.ended.update(self.batches)


def _requests(*ids: str, prompt: str = "go") -> list[BatchRequest]:
    return [BatchRequest(request_id, LLMMessage.system("rules"), [LLMMessage.user(f"{prompt} {request_id}")]) for request_id in ids]


def _batch(
    client: _FakeBatchClient,
    store: JsonBatchStore,
    requests: list[BatchRequest],
    model_name: str = "stub",
    options: LLMCallOptions | None = None,
    output_schema: type[BaseModel] | None = None,
    wait: bool = False,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> BatchRun[_StubModel, str]:
    return BatchRun(client, model_name, options or LLMCallOptions(), requests, store, output_schema, wait, poll_interval)


async def _run(client: _FakeBatchClient, store: JsonBatchStore, requests: list[BatchRequest], **kwargs: Any):
    async with _batch(client, store, requests, **kwargs) as outcome:
        return outcome


@pytest.fixture
def store(tmp_path: Path) -> JsonBatchStore:
    return JsonBatchStore(tmp_path / "batches.json")


# --- one run after another ----------------------------------------------------------------------------


def test_new_requests_are_submitted_recorded_and_pending(store: JsonBatchStore):
    client = _FakeBatchClient()

    outcome = asyncio.run(_run(client, store, _requests("a", "b")))

    assert outcome.results == {} and outcome.pending == {"a", "b"}
    assert [body for _, body in client.batches["batch-0"]] == [{"system": "rules", "prompt": "go a"}, {"system": "rules", "prompt": "go b"}]
    [job] = store.load()
    assert job.batch_id == "batch-0" and job.provider is Provider.OPENAI and job.model == "openai:stub" and list(job.requests) == ["a", "b"]


def test_a_second_run_does_not_resubmit_running_requests(store: JsonBatchStore):
    client = _FakeBatchClient()
    asyncio.run(_run(client, store, _requests("a", "b")))

    outcome = asyncio.run(_run(client, store, _requests("a", "b", "c")))

    assert list(client.batches) == ["batch-0", "batch-1"] and [request_id for request_id, _ in client.batches["batch-1"]] == ["c"]
    assert outcome.pending == {"a", "b", "c"}


def test_an_ended_job_is_collected_parsed_and_marked_as_batch(store: JsonBatchStore):
    client = _FakeBatchClient()
    asyncio.run(_run(client, store, _requests("a", "b")))
    client.end_all()

    outcome = asyncio.run(_run(client, store, _requests("a", "b")))

    assert outcome.pending == set()
    answer = outcome.results["a"]
    assert isinstance(answer, LLMResponse) and answer.output_text == "go a" and answer.batch


def test_a_clean_exit_releases_the_collected_jobs(store: JsonBatchStore):
    client = _FakeBatchClient()
    asyncio.run(_run(client, store, _requests("a")))
    client.end_all()
    asyncio.run(_run(client, store, _requests("a")))

    assert store.load() == []
    # released ids are new requests again
    asyncio.run(_run(client, store, _requests("a")))
    assert list(client.batches) == ["batch-0", "batch-1"]


def test_an_exception_in_the_block_keeps_the_jobs_so_the_next_run_collects_them_again(store: JsonBatchStore):
    client = _FakeBatchClient()
    asyncio.run(_run(client, store, _requests("a")))
    client.end_all()

    async def crash_while_saving() -> None:
        async with _batch(client, store, _requests("a")) as outcome:
            assert "a" in outcome.results
            raise OSError("disk full")

    with pytest.raises(OSError):
        asyncio.run(crash_while_saving())

    assert [job.batch_id for job in store.load()] == ["batch-0"]
    outcome = asyncio.run(_run(client, store, _requests("a")))
    assert isinstance(outcome.results["a"], LLMResponse) and list(client.batches) == ["batch-0"]


def test_a_collected_job_returns_ids_this_run_did_not_ask_for(store: JsonBatchStore):
    # the job leaves the store as a whole, so holding these back would lose them
    client = _FakeBatchClient()
    asyncio.run(_run(client, store, _requests("a", "b")))
    client.end_all()

    outcome = asyncio.run(_run(client, store, _requests("a")))

    assert set(outcome.results) == {"a", "b"}


def test_new_requests_are_split_at_the_provider_byte_limit(store: JsonBatchStore):
    client = _FakeBatchClient()
    client.max_batch_requests = 100
    client.max_batch_bytes = 10
    # prompts of 4, 4, 4 and 30 bytes: two fit under 10, and the one over the limit still gets its own batch
    requests = [*_requests("a", "b", "c"), BatchRequest("long", LLMMessage.system("rules"), [LLMMessage.user("x" * 30)])]

    asyncio.run(_run(client, store, requests))

    assert [[request_id for request_id, _ in batch] for batch in client.batches.values()] == [["a", "b"], ["c"], ["long"]]


def test_new_requests_are_split_at_the_provider_limit(store: JsonBatchStore):
    client = _FakeBatchClient()

    asyncio.run(_run(client, store, _requests("a", "b", "c", "d", "e")))

    assert [[request_id for request_id, _ in batch] for batch in client.batches.values()] == [["a", "b"], ["c", "d"], ["e"]]
    assert [job.batch_id for job in store.load()] == ["batch-0", "batch-1", "batch-2"]


def test_wait_polls_until_every_job_has_ended(store: JsonBatchStore):
    client = _FakeBatchClient()
    client.end_after_checks = 5  # two jobs: ended on the third round of checks

    outcome = asyncio.run(_run(client, store, _requests("a", "b", "c"), wait=True, poll_interval=0.001))

    assert outcome.pending == set() and set(outcome.results) == {"a", "b", "c"}


def test_without_wait_the_jobs_are_checked_once(store: JsonBatchStore):
    client = _FakeBatchClient()
    client.end_after_checks = 5

    outcome = asyncio.run(_run(client, store, _requests("a", "b", "c")))

    assert client.status_checks == 2 and outcome.pending == {"a", "b", "c"}


# --- what each request ends as --------------------------------------------------------------------------


def test_failed_expired_cancelled_and_missing_requests_each_keep_their_kind(store: JsonBatchStore):
    client = _FakeBatchClient()
    asyncio.run(_run(client, store, _requests("ok", "bad")))
    asyncio.run(_run(client, store, _requests("late", "stopped")))
    asyncio.run(_run(client, store, _requests("lost")))
    client.answers = {
        "bad": LLMCallError(FailureKind.PROVIDER, "invalid_request_error"),
        "late": LLMCallError(FailureKind.EXPIRED, "expired"),
        "stopped": LLMCallError(FailureKind.CANCELLED, "cancelled"),
        "lost": None,
    }
    client.end_all()

    outcome = asyncio.run(_run(client, store, _requests("ok", "bad", "late", "stopped", "lost")))

    kinds = {request_id: result.kind for request_id, result in outcome.results.items() if isinstance(result, LLMCallError)}
    assert kinds == {"bad": FailureKind.PROVIDER, "late": FailureKind.EXPIRED, "stopped": FailureKind.CANCELLED, "lost": FailureKind.PROVIDER}
    assert isinstance(outcome.results["ok"], LLMResponse)


def test_an_answer_off_the_schema_keeps_its_response_priced_as_batch(store: JsonBatchStore):
    client = _FakeBatchClient()
    asyncio.run(_run(client, store, _requests("a"), output_schema=_Move))
    client.answers = {"a": '{"move": "centre"}'}
    client.end_all()

    outcome = asyncio.run(_run(client, store, _requests("a"), output_schema=_Move))

    error = outcome.results["a"]
    assert isinstance(error, LLMCallError) and error.kind is FailureKind.UNPARSEABLE
    assert error.response is not None and error.response.batch


# --- what stops a run -------------------------------------------------------------------------------------


def test_the_same_id_with_another_prompt_is_refused(store: JsonBatchStore):
    client = _FakeBatchClient()
    asyncio.run(_run(client, store, _requests("a")))

    with pytest.raises(ValueError, match="another prompt or schema"):
        asyncio.run(_run(client, store, _requests("a", prompt="changed")))


def test_the_same_ids_run_for_several_model_configs_in_one_store(store: JsonBatchStore):
    # comparing models on the same items: each model config has its own ids
    client = _FakeBatchClient()
    asyncio.run(_run(client, store, _requests("a", "b")))
    asyncio.run(_run(client, store, _requests("a", "b"), options=LLMCallOptions(max_tokens=100)))

    assert [(job.model, list(job.requests)) for job in store.load()] == [("openai:stub", ["a", "b"]), ("openai:stub:max_tokens=100", ["a", "b"])]
    client.end_all()
    outcome = asyncio.run(_run(client, store, _requests("a", "b"), options=LLMCallOptions(max_tokens=100)))

    # collecting one model config leaves the other's job running
    assert set(outcome.results) == {"a", "b"} and [job.model for job in store.load()] == ["openai:stub"]


def test_the_request_hash_covers_the_model_config_too():
    # ids are already matched within one model config; the hash still names the model, so it alone says what was asked
    [request] = _requests("a")

    assert request_hash("openai:stub", request, None) != request_hash("openai:stub:max_tokens=100", request, None)


def test_the_same_id_with_another_schema_is_refused(store: JsonBatchStore):
    client = _FakeBatchClient()
    asyncio.run(_run(client, store, _requests("a")))

    with pytest.raises(ValueError, match="another prompt or schema"):
        asyncio.run(_run(client, store, _requests("a"), output_schema=_Move))


def test_a_submission_that_timed_out_keeps_its_record_and_stops_the_next_run(store: JsonBatchStore):
    client = _FakeBatchClient()
    client.submit_error = TimeoutError("read timed out")

    with pytest.raises(LLMCallError) as caught:
        asyncio.run(_run(client, store, _requests("a")))
    assert caught.value.kind is FailureKind.TIMEOUT

    [job] = store.load()
    assert job.batch_id is None
    client.submit_error = None
    with pytest.raises(RuntimeError, match="never confirmed"):
        asyncio.run(_run(client, store, _requests("a")))
    assert client.batches == {}


def test_a_submission_the_provider_rejected_leaves_no_record(store: JsonBatchStore):
    client = _FakeBatchClient()
    client.submit_error = _StatusError("400 invalid file")

    with pytest.raises(LLMCallError) as caught:
        asyncio.run(_run(client, store, _requests("a")))

    assert caught.value.kind is FailureKind.PROVIDER and store.load() == []


def test_a_rejected_chunk_keeps_the_chunks_submitted_before_it(store: JsonBatchStore):
    client = _FakeBatchClient()
    real_submit = client._submit_batch

    async def fail_second(model: _StubModel, requests: list[tuple[str, dict[str, Any]]]) -> str:
        if client.batches:
            raise _StatusError("429 too many batches")
        return await real_submit(model, requests)

    client._submit_batch = fail_second  # type: ignore[method-assign]

    with pytest.raises(LLMCallError):
        asyncio.run(_run(client, store, _requests("a", "b", "c")))

    assert [(job.batch_id, list(job.requests)) for job in store.load()] == [("batch-0", ["a", "b"])]


def test_checks_before_anything_is_sent(store: JsonBatchStore):
    client = _FakeBatchClient()

    with pytest.raises(ValueError, match="unique"):
        _batch(client, store, _requests("a", "a"))
    with pytest.raises(ValueError, match="no batch price"):
        _batch(client, store, _requests("a"), model_name="no-batch-price")
    with pytest.raises(ValueError, match="thinking"):
        _batch(client, store, _requests("a"), options=LLMCallOptions(thinking_enabled=True))
    with pytest.raises(ValueError, match="poll_interval"):
        _batch(client, store, _requests("a"), poll_interval=0)
    assert client.batches == {} and not Path(store._path).exists()


def test_a_provider_without_batch_support_says_so(store: JsonBatchStore):
    client = _FakeBatchClient()
    client.supports_batch = False

    with pytest.raises(NotImplementedError, match="batches are not supported"):
        _batch(client, store, _requests("a"))


def test_cancel_asks_the_provider_to_stop_its_running_jobs_and_keeps_them_in_the_store(store: JsonBatchStore):
    client = _FakeBatchClient()
    asyncio.run(_run(client, store, _requests("a", "b", "c")))

    asyncio.run(cancel_batches(client, store))

    assert client.cancelled == ["batch-0", "batch-1"] and len(store.load()) == 2


def test_the_router_hands_the_model_config_to_the_provider(store: JsonBatchStore):
    client = _FakeBatchClient()
    llm = LLM(LLMConfig())
    llm._provider_map[Provider.OPENAI] = client

    async def run() -> None:
        async with llm.batch(ModelConfig(Provider.OPENAI, "stub"), _requests("a"), store) as outcome:
            assert outcome.pending == {"a"}

    asyncio.run(run())
    assert store.load()[0].requests["a"] == request_hash("openai:stub", _requests("a")[0], None)


# --- the store ----------------------------------------------------------------------------------------------


def test_the_store_round_trips_and_leaves_no_temporary_file(store: JsonBatchStore, tmp_path: Path):
    job = BatchJob("claude:claude-opus-5:thinking_enabled=True", "msgbatch_1", "2026-10-11T10:00:00+00:00", {"a": "h1", "b": "h2"})

    store.save([job])

    assert store.load() == [job]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["batches.json"]
    stored = json.loads((tmp_path / "batches.json").read_text())
    assert stored["version"] == 2
    # the provider is read off the model config, not stored twice
    assert set(stored["jobs"][0]) == {"model", "batch_id", "submitted_at", "requests"} and job.provider is Provider.CLAUDE


def test_a_store_of_another_version_is_refused(store: JsonBatchStore, tmp_path: Path):
    (tmp_path / "batches.json").write_text(json.dumps({"version": 99, "jobs": []}))

    with pytest.raises(ValueError, match="version"):
        store.load()


def test_the_lock_admits_one_holder_at_a_time(store: JsonBatchStore):
    # two coroutines of one process: a blocking flock would stall the loop and never let the holder finish
    inside: list[int] = []
    overlaps: list[bool] = []

    async def hold(name: int) -> None:
        async with store.lock():
            inside.append(name)
            overlaps.append(len(inside) > 1)
            await asyncio.sleep(0.02)
            inside.remove(name)

    async def both() -> None:
        await asyncio.gather(hold(1), hold(2))

    asyncio.run(both())

    assert overlaps == [False, False]
