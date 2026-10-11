from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from plyllm.errors import LLMCallError
from plyllm.message import LLMMessage
from plyllm.model_config import ModelConfig
from plyllm.providers.providers import Provider
from plyllm.response import LLMResponse

# seconds between two status checks while `wait=True`; a batch takes minutes to hours, so checking once a minute costs nothing and adds at most a minute to the wait
DEFAULT_POLL_INTERVAL = 60.0
# seconds between two attempts to take a store lock another run or coroutine holds
_LOCK_RETRY = 0.1
_STORE_VERSION = 2


@dataclass(frozen=True)
class BatchRequest:
    # the caller's key for this request, unique within one call; results come back under it
    id: str
    system: LLMMessage
    messages: list[LLMMessage]


@dataclass(frozen=True)
class BatchJob:
    # one provider batch, as the store records it
    # the full model config string (provider:model:options); request ids are unique per model config, so one store
    # can hold the same ids for several models
    model: str
    # None between writing the record and the provider confirming the batch; see BatchRun._submit
    batch_id: str | None
    submitted_at: str  # ISO 8601, UTC
    # id -> request hash, in submission order (Gemini maps its results back by position)
    requests: dict[str, str]

    @property
    def provider(self) -> Provider:
        return ModelConfig.from_string(self.model).provider

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "batch_id": self.batch_id, "submitted_at": self.submitted_at, "requests": self.requests}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchJob:
        return cls(data["model"], data["batch_id"], data["submitted_at"], dict(data["requests"]))


@dataclass(frozen=True)
class BatchOutcome:
    # every id of every job that has ended, including ids the caller did not pass this time: a job leaves the
    # store as a whole, so answers held back now would be lost
    results: dict[str, LLMResponse | LLMCallError] = field(default_factory=dict)
    # every id of the jobs still running
    pending: set[str] = field(default_factory=set)


class BatchStore(Protocol):
    def lock(self) -> AbstractAsyncContextManager[None]:
        # exclusive across processes and across coroutines of one process
        ...

    def load(self) -> list[BatchJob]: ...

    def save(self, jobs: list[BatchJob]) -> None:
        # replaces every record, atomically
        ...


class JsonBatchStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock_path = self._path.with_name(f"{self._path.name}.lock")

    @asynccontextmanager
    async def lock(self) -> AsyncGenerator[None]:
        import fcntl  # POSIX only, so imported here: importing plyllm must still work elsewhere

        with open(self._lock_path, "a") as handle:
            # non-blocking with a retry, since a blocking flock would stall the event loop, and with it the
            # coroutine of this process that holds the lock
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(_LOCK_RETRY)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def load(self) -> list[BatchJob]:
        if not self._path.exists():
            return []
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if data.get("version") != _STORE_VERSION:
            raise ValueError(f"{self._path} has store version {data.get('version')!r}; this plyllm reads version {_STORE_VERSION}")
        return [BatchJob.from_dict(job) for job in data["jobs"]]

    def save(self, jobs: list[BatchJob]) -> None:
        temporary = self._path.with_name(f"{self._path.name}.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"version": _STORE_VERSION, "jobs": [job.to_dict() for job in jobs]}, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._path)


def request_hash(model_config: str, request: BatchRequest, output_schema: type[BaseModel] | None) -> str:
    # the whole request, independent of any SDK's request objects (Gemini's and Mistral's are not JSON). Ids are
    # matched within one model config already; the model is hashed too, so a hash alone says what was asked.
    # The same id with another hash is another question, which must not get the answer to the first
    payload = {
        "model": model_config,
        "system": request.system.to_dict(),
        "messages": [message.to_dict() for message in request.messages],
        "schema": output_schema.model_json_schema() if output_schema is not None else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
