import asyncio

import pytest

from plybench.llm.rate_limit import ModelGate, ModelLimits, NoLimits, estimate_prompt_tokens, make_gate


class FakeClock:
    """Virtual time: sleeps advance the clock instead of waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.slept.append(delay)
        self.now += delay
        # yield so other tasks get a turn, matching real asyncio.sleep scheduling
        await asyncio.sleep(0)


def _gate(clock: FakeClock, **limits) -> ModelGate:
    return ModelGate(ModelLimits(**limits), clock=clock.time, sleep=clock.sleep)


async def _noop() -> int:
    return 0


def test_no_limits_passes_through():
    gate = NoLimits()
    assert asyncio.run(gate.run(lambda: _noop())) == 0


def test_make_gate_returns_no_limits_when_nothing_is_enforced():
    assert isinstance(make_gate(None), NoLimits)
    assert isinstance(make_gate(ModelLimits()), NoLimits)
    assert isinstance(make_gate(ModelLimits(tpm=1000)), ModelGate)


def test_requests_are_paced_to_the_configured_rate():
    clock = FakeClock()
    gate = _gate(clock, rps=2.0)  # one request every 0.5s

    async def scenario() -> None:
        for _ in range(3):
            await gate.run(lambda: _noop())

    asyncio.run(scenario())

    # the first request goes immediately, the next two are spaced by the interval
    assert clock.slept == pytest.approx([0.5, 0.5])
    assert clock.now == pytest.approx(1.0)


def test_token_window_blocks_until_the_oldest_reservation_expires():
    clock = FakeClock()
    gate = _gate(clock, tpm=1000)

    async def scenario() -> None:
        await gate.run(lambda: _noop(), estimate=800)
        # 800 + 400 exceeds the window, so this waits out the first reservation
        await gate.run(lambda: _noop(), estimate=400)

    asyncio.run(scenario())

    assert clock.slept == pytest.approx([60.0])


def test_settling_actual_usage_frees_the_overestimate():
    clock = FakeClock()
    gate = _gate(clock, tpm=1000)

    async def scenario() -> None:
        # reserve 800 but only use 100, so the follow-up fits without waiting
        await gate.run(lambda: _noop(), estimate=800, tokens_of=lambda _: 100)
        await gate.run(lambda: _noop(), estimate=400)

    asyncio.run(scenario())

    assert clock.slept == []


def test_failed_call_keeps_its_estimate_reserved():
    clock = FakeClock()
    gate = _gate(clock, tpm=1000)

    async def boom() -> int:
        raise RuntimeError("api failure")

    async def scenario() -> None:
        with pytest.raises(RuntimeError):
            await gate.run(boom, estimate=800, tokens_of=lambda _: 100)
        await gate.run(lambda: _noop(), estimate=400)

    asyncio.run(scenario())

    # the estimate was never settled, so the window still counts the full 800
    assert clock.slept == pytest.approx([60.0])


def test_request_larger_than_the_whole_window_is_admitted_alone():
    clock = FakeClock()
    gate = _gate(clock, tpm=1000)

    asyncio.run(gate.run(lambda: _noop(), estimate=5000))

    assert clock.slept == []


def test_expired_reservations_stop_counting():
    clock = FakeClock()
    gate = _gate(clock, tpm=1000)

    async def scenario() -> None:
        await gate.run(lambda: _noop(), estimate=900)
        clock.now += 61.0  # the reservation ages out of the window
        await gate.run(lambda: _noop(), estimate=900)

    asyncio.run(scenario())

    assert clock.slept == []


def test_concurrency_share_caps_parallel_calls():
    clock = FakeClock()
    gate = _gate(clock, max_concurrent=2)
    in_flight = 0
    peak = 0

    async def tracked() -> int:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return 0

    async def scenario() -> None:
        await asyncio.gather(*(gate.run(tracked) for _ in range(6)))

    asyncio.run(scenario())

    assert peak == 2


def test_scaled_leaves_concurrency_untouched():
    scaled = ModelLimits(max_concurrent=8, rps=10.0, tpm=100_000).scaled(0.5)

    assert scaled.max_concurrent == 8
    assert scaled.rps == pytest.approx(5.0)
    assert scaled.tpm == 50_000


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        ModelLimits(max_concurrent=0)
    with pytest.raises(ValueError):
        ModelLimits(rps=0)
    with pytest.raises(ValueError):
        ModelLimits(tpm=0)
    with pytest.raises(ValueError):
        ModelLimits(tpm=100).scaled(0)


def test_prompt_estimate_counts_every_message():
    assert estimate_prompt_tokens("a" * 40, "b" * 40) == 20
