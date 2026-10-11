from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Generic

from pydantic import BaseModel

from plyllm.batch import BatchJob, BatchOutcome, BatchRequest, BatchStore, request_hash
from plyllm.client import LLMClient, ModelT, ResponseT
from plyllm.errors import FailureKind, LLMCallError, as_call_error
from plyllm.model_config import ModelConfig
from plyllm.options import LLMCallOptions
from plyllm.response import LLMResponse


class BatchRun(Generic[ModelT, ResponseT]):
    def __init__(
        self,
        client: LLMClient[ModelT, ResponseT],
        model_name: str,
        options: LLMCallOptions,
        requests: Sequence[BatchRequest],
        store: BatchStore,
        output_schema: type[BaseModel] | None,
        wait: bool,
        poll_interval: float,
    ) -> None:
        # everything that can fail before a request is sent fails here, before the block is entered
        if not client.supports_batch:
            raise NotImplementedError(f"{client.provider_key.value} batches are not supported in this package")
        model = client._resolve_for_generate(model_name, output_schema)
        model.validate(options)
        if model.batch_ratio is None:
            raise ValueError(f"Model {model.model_name} has no batch price, so its batch answers could not be costed")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        ids = [request.id for request in requests]
        duplicates = sorted({request_id for request_id in ids if ids.count(request_id) > 1})
        if duplicates:
            raise ValueError(f"Batch request ids must be unique within one call; repeated: {duplicates}")
        for request_id in ids:
            client._check_batch_id(request_id)

        self._client = client
        self._model = model
        self._options = options
        self._requests = list(requests)
        self._store = store
        self._output_schema = output_schema
        self._wait = wait
        self._poll_interval = poll_interval
        self._collected: set[str] = set()

    async def __aenter__(self) -> BatchOutcome:
        jobs = await self._submit()
        ended = await self._poll(jobs)
        results: dict[str, LLMResponse | LLMCallError] = {}
        for job in ended:
            results.update(await self._collect(job))
        self._collected = {job.batch_id for job in ended if job.batch_id is not None}
        pending = {request_id for job in jobs if job not in ended for request_id in job.requests}
        return BatchOutcome(results, pending)

    async def __aexit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        if exc_type is not None or not self._collected:
            return
        async with self._store.lock():
            self._store.save([job for job in self._store.load() if job.batch_id not in self._collected])

    async def _submit(self) -> list[BatchJob]:
        # the jobs that hold this run's requests, after submitting the new ones. Under the lock, so two runs
        # cannot both find an id new and pay for it twice
        client, model = self._client, self._model
        model_config = ModelConfig(client.provider_key, model.model_name, self._options).to_string()
        hashes = {request.id: request_hash(model_config, request, self._output_schema) for request in self._requests}

        async with self._store.lock():
            stored = self._store.load()
            holder = {request_id: job for job in stored for request_id in job.requests}
            tracked: dict[str, BatchJob] = {}
            new: list[BatchRequest] = []
            for request in self._requests:
                job = holder.get(request.id)
                if job is None:
                    new.append(request)
                elif job.requests[request.id] != hashes[request.id]:
                    raise ValueError(f"Request {request.id} is already in a running batch with another prompt, model or schema; use a new id, or wait for that batch")
                elif job.batch_id is None:
                    raise RuntimeError(
                        f"Request {request.id} is in a batch whose submission never confirmed (submitted {job.submitted_at}). The provider may have "
                        f"created it: check its batch list, then remove that record from the store or fill in its batch_id"
                    )
                else:
                    tracked[job.batch_id] = job

            for start in range(0, len(new), client.max_batch_requests):
                chunk = new[start : start + client.max_batch_requests]
                # built before the record is written, so a request the model rejects leaves nothing behind
                bodies = [(request.id, client._request(model, request.system, request.messages, self._options, self._output_schema)) for request in chunk]
                record = BatchJob(client.provider_key, model.model_name, None, _now(), {request.id: hashes[request.id] for request in chunk})
                stored.append(record)
                self._store.save(stored)
                try:
                    batch_id = await client._submit_batch(model, bodies)
                except Exception as error:
                    kind = client.error_kind(error)
                    if kind in (FailureKind.TIMEOUT, FailureKind.CONNECTION):
                        # the provider may have created the batch without us hearing back: keep the record, so
                        # the next run stops instead of paying for these requests twice
                        error.add_note("The batch may exist at the provider; the store keeps its record without a batch_id until checked by hand.")
                    else:
                        stored.remove(record)
                        self._store.save(stored)
                    if kind is FailureKind.OTHER:
                        raise  # not the provider's answer, e.g. a ValueError for an input over its limits
                    raise as_call_error(error, kind) from error
                submitted = replace(record, batch_id=batch_id)
                stored[stored.index(record)] = submitted
                self._store.save(stored)
                tracked[batch_id] = submitted
        return list(tracked.values())

    async def _poll(self, jobs: list[BatchJob]) -> list[BatchJob]:
        ended: list[BatchJob] = []
        while True:
            for job in jobs:
                if job not in ended and await self._client._batch_api(lambda job=job: self._client._batch_ended(_batch_id(job))):
                    ended.append(job)
            if not self._wait or len(ended) == len(jobs):
                return ended
            await asyncio.sleep(self._poll_interval)

    async def _collect(self, job: BatchJob) -> dict[str, LLMResponse | LLMCallError]:
        raw = await self._client._batch_api(lambda: self._client._batch_results(job))
        results: dict[str, LLMResponse | LLMCallError] = {}
        for request_id in job.requests:
            item = raw.get(request_id)
            if item is None:
                results[request_id] = LLMCallError(FailureKind.PROVIDER, f"Request {request_id} is missing from the results of batch {job.batch_id}")
            elif isinstance(item, LLMCallError):
                results[request_id] = item
            else:
                try:
                    results[request_id] = _as_batch(self._client._parse(self._model, item, self._output_schema))
                except LLMCallError as error:
                    results[request_id] = _as_batch(error)
        return results


async def cancel_batches(client: LLMClient[ModelT, ResponseT], store: BatchStore) -> None:
    # asks the provider to stop this provider's running jobs; the store keeps them, and the next run collects
    # them as ended, with the requests that never ran failing as CANCELLED
    async with store.lock():
        jobs = store.load()
    for batch_id in [job.batch_id for job in jobs if job.provider is client.provider_key and job.batch_id is not None]:
        await client._batch_api(lambda batch_id=batch_id: client._cancel_batch(batch_id))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _as_batch(result: LLMResponse | LLMCallError) -> LLMResponse | LLMCallError:
    # a batch answer is billed at the batch price; the flag keeps it from being costed at the standard one
    if isinstance(result, LLMResponse):
        return replace(result, batch=True)
    if result.response is not None:
        result.response = replace(result.response, batch=True)
    return result


def _batch_id(job: BatchJob) -> str:
    if job.batch_id is None:
        raise RuntimeError("a job without a batch_id is never polled")  # _submit stops on those
    return job.batch_id
