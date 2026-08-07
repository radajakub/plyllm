from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Protocol, TypeVar

T = TypeVar("T")

# the token allowance providers publish is per minute, so the accounting window is fixed at 60s
WINDOW_SECONDS = 60.0
# rough chars-per-token ratio; only used to size a reservation before the API reports real usage
CHARS_PER_TOKEN = 4
# how long a blocked token reservation waits before re-checking, when the window gives no better hint
_MIN_RETRY_DELAY = 0.01

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class ModelLimits:
    # every field is independently optional: unset means that dimension is not enforced
    max_concurrent: int | None = None
    rps: float | None = None
    tpm: int | None = None

    def __post_init__(self) -> None:
        if self.max_concurrent is not None and self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if self.rps is not None and self.rps <= 0:
            raise ValueError("rps must be positive")
        if self.tpm is not None and self.tpm < 1:
            raise ValueError("tpm must be at least 1")

    @property
    def enforced(self) -> bool:
        return self.max_concurrent is not None or self.rps is not None or self.tpm is not None

    def scaled(self, factor: float) -> ModelLimits:
        # provider quotas are org-wide while a gate only sees its own process; scaling down leaves
        # headroom for concurrent runs. concurrency is a local knob, so it is deliberately untouched.
        if factor <= 0:
            raise ValueError("Rate limit scale must be positive")

        scaled_rps = self.rps
        if scaled_rps is not None:
            scaled_rps *= factor

        scaled_tpm = self.tpm
        if scaled_tpm is not None:
            scaled_tpm = max(int(scaled_tpm * factor), 1)

        return ModelLimits(max_concurrent=self.max_concurrent, rps=scaled_rps, tpm=scaled_tpm)


def estimate_prompt_tokens(*contents: str) -> int:
    return sum(len(content) for content in contents) // CHARS_PER_TOKEN


@dataclass
class _Reservation:
    at: float
    tokens: int


# base class for rate limiters
class RateGate(Protocol):
    async def run(self, task: Callable[[], Awaitable[T]], estimate: int = 0, tokens_of: Callable[[T], int] | None = None) -> T: ...


# without limits, just run the task
class NoLimits(RateGate):
    async def run(self, task: Callable[[], Awaitable[T]], estimate: int = 0, tokens_of: Callable[[T], int] | None = None) -> T:
        return await task()


# a gate that enforces limits for a given model
class ModelGate(RateGate):
    def __init__(self, limits: ModelLimits, clock: Clock = monotonic, sleep: Sleeper = asyncio.sleep) -> None:
        self._limits = limits
        # injectable so tests can drive the windows without real waiting
        self._clock = clock
        self._sleep = sleep

        self._semaphore = asyncio.Semaphore(limits.max_concurrent) if limits.max_concurrent is not None else None
        # earliest time the next request may start; paces arrivals evenly instead of in bursts
        self._next_slot = 0.0
        self._slot_lock = asyncio.Lock()
        self._reservations: list[_Reservation] = []
        self._token_lock = asyncio.Lock()

    @property
    def limits(self) -> ModelLimits:
        return self._limits

    async def run(self, task: Callable[[], Awaitable[T]], estimate: int = 0, tokens_of: Callable[[T], int] | None = None) -> T:
        if self._semaphore is None:
            return await self._paced(task, estimate, tokens_of)

        async with self._semaphore:
            return await self._paced(task, estimate, tokens_of)

    async def _paced(self, task: Callable[[], Awaitable[T]], estimate: int, tokens_of: Callable[[T], int] | None) -> T:
        await self._await_slot()
        reservation = await self._reserve_tokens(estimate)

        result = await task()

        # a failed call keeps its estimate reserved, which errs towards throttling rather than 429s
        if reservation is not None and tokens_of is not None:
            reservation.tokens = max(tokens_of(result), 0)
        return result

    async def _await_slot(self) -> None:
        rps = self._limits.rps
        if rps is None:
            return

        interval = 1.0 / rps
        async with self._slot_lock:
            # claim the next slot under the lock, then wait for it outside so waiters queue in order
            # without holding up the queue while they sleep
            start = max(self._clock(), self._next_slot)
            self._next_slot = start + interval

        delay = start - self._clock()
        if delay > 0:
            await self._sleep(delay)

    async def _reserve_tokens(self, estimate: int) -> _Reservation | None:
        limit = self._limits.tpm
        if limit is None:
            return None

        wanted = max(estimate, 0)
        while True:
            async with self._token_lock:
                now = self._clock()
                self._drop_expired(now)
                used = sum(reservation.tokens for reservation in self._reservations)

                # an empty window always admits: a single request larger than the whole minute's
                # allowance would otherwise wait forever
                if not self._reservations or used + wanted <= limit:
                    reservation = _Reservation(now, wanted)
                    self._reservations.append(reservation)
                    return reservation

                # the window frees up when its oldest entry ages out
                delay = WINDOW_SECONDS - (now - self._reservations[0].at)

            await self._sleep(max(delay, _MIN_RETRY_DELAY))

    def _drop_expired(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        self._reservations = [reservation for reservation in self._reservations if reservation.at > cutoff]


# construct a gate from limits
def make_gate(limits: ModelLimits | None) -> RateGate:
    if limits is None or not limits.enforced:
        return NoLimits()
    return ModelGate(limits)
