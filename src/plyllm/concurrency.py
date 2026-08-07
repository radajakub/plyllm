from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class ProviderSemaphore:
    def __init__(self, concurrency: int = 10) -> None:
        if concurrency < 1:
            raise ValueError("Concurrency must be at least 1")
        self._concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

    @property
    def concurrency(self) -> int:
        return self._concurrency

    def configure(self, concurrency: int | None) -> None:
        if concurrency is None:
            return
        if concurrency < 1:
            raise ValueError("Concurrency must be at least 1")

        self._concurrency = concurrency
        # in-flight tasks keep the old semaphore; new acquisitions use the new limit
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run(self, task: Callable[[], Awaitable[T]]) -> T:
        async with self._semaphore:
            return await task()


async def safe_call(
    task: Callable[[], Awaitable[T]],
    retry_errors: tuple[type[Exception], ...],
    retries: int = 10,
    delay_base: float = 2.0,
    delay_max: float = 30.0,
    retry_if: Callable[[Exception], bool] | None = None,
) -> T:
    # retry_if narrows retry_errors for SDKs that funnel every HTTP failure into one exception type,
    # where only some statuses (throttling, 5xx) are worth another attempt
    for attempt in range(retries):
        try:
            return await task()
        except retry_errors as error:
            if attempt == retries - 1 or (retry_if is not None and not retry_if(error)):
                raise
            delay = min(delay_base * (2**attempt), delay_max) * random.uniform(0.7, 1.2)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable: safe_call exhausted retries without returning or raising")
