"""Concurrency permits: the caps are never exceeded, and they are always released.

SPEC NFR puts "correctness under concurrency" first, so these tests instrument an actual
counter rather than trusting the semaphore. Every one of them forces genuine overlap: a
task that holds its slot until released by the test, so a cap that silently did nothing
would show up as a peak above the limit rather than as a passing test.
"""

import asyncio

import pytest

from dagent.errors import PolicyError
from dagent.policy.limits import Limits


class Counter:
    """Counts how many tasks are inside a slot at once."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0
        self.completed = 0

    async def occupy(self, limits: Limits, provider: str, hold: asyncio.Event) -> None:
        async with limits.slot(provider):
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            await hold.wait()
            self.in_flight -= 1
            self.completed += 1


async def drain(limits: Limits, provider: str, *, tasks: int, counter: Counter) -> None:
    """Start `tasks` occupants, let them through one release at a time, and wait."""
    hold = asyncio.Event()
    running = [asyncio.create_task(counter.occupy(limits, provider, hold)) for _ in range(tasks)]
    # Let everything that can get a permit take one before anything is allowed to finish;
    # that is what makes `peak` the true measure of simultaneity.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    hold.set()
    await asyncio.gather(*running)


@pytest.mark.asyncio
async def test_limits_default_to_unbounded() -> None:
    # A run that asks for no cap must behave exactly as it did before caps existed.
    counter = Counter()

    await drain(Limits(), "stub", tasks=8, counter=counter)

    assert counter.peak == 8


@pytest.mark.asyncio
async def test_the_global_cap_is_never_exceeded() -> None:
    counter = Counter()

    await drain(Limits(max_concurrency=2), "stub", tasks=6, counter=counter)

    assert counter.peak == 2
    assert counter.completed == 6


@pytest.mark.asyncio
async def test_the_per_provider_cap_is_never_exceeded() -> None:
    counter = Counter()

    await drain(Limits(per_provider={"gemini": 3}), "gemini", tasks=7, counter=counter)

    assert counter.peak == 3


@pytest.mark.asyncio
async def test_an_unlisted_provider_gets_the_default_cap() -> None:
    counter = Counter()
    limits = Limits(per_provider={"gemini": 5}, default_per_provider=1)

    await drain(limits, "somebody-else", tasks=4, counter=counter)

    assert counter.peak == 1


@pytest.mark.asyncio
async def test_providers_are_capped_independently_of_one_another() -> None:
    # DR-6: one provider's slowness must not starve another past the global cap.
    limits = Limits(per_provider={"slow": 1, "fast": 3})
    slow, fast = Counter(), Counter()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(slow.occupy(limits, "slow", hold)) for _ in range(3)]
    tasks += [asyncio.create_task(fast.occupy(limits, "fast", hold)) for _ in range(3)]
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert slow.in_flight == 1
    assert fast.in_flight == 3

    hold.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_the_tighter_of_the_two_axes_wins() -> None:
    # Both permits are required, so the effective cap is the smaller one.
    counter = Counter()
    limits = Limits(max_concurrency=4, per_provider={"gemini": 2})

    await drain(limits, "gemini", tasks=6, counter=counter)

    assert counter.peak == 2


@pytest.mark.asyncio
async def test_the_global_cap_binds_across_providers() -> None:
    limits = Limits(max_concurrency=2, per_provider={"a": 5, "b": 5})
    counter = Counter()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(counter.occupy(limits, "a", hold)) for _ in range(3)]
    tasks += [asyncio.create_task(counter.occupy(limits, "b", hold)) for _ in range(3)]
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert counter.peak == 2

    hold.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_a_permit_is_released_when_the_body_raises() -> None:
    # Released in `finally`, or one exception permanently shrinks the pool.
    limits = Limits(max_concurrency=1)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            async with limits.slot("stub"):
                raise RuntimeError("boom")

    async with limits.slot("stub"):
        assert limits.in_flight("stub") == 1


@pytest.mark.asyncio
async def test_a_permit_is_released_when_the_body_is_cancelled() -> None:
    # A node cancelled by its timeout must not take its permits to the grave.
    limits = Limits(max_concurrency=1)
    entered = asyncio.Event()

    async def hang() -> None:
        async with limits.slot("stub"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hang())
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The permit came back: this would block forever otherwise.
    async with asyncio.timeout(5), limits.slot("stub"):
        pass


@pytest.mark.asyncio
async def test_in_flight_returns_to_zero_and_peak_remembers() -> None:
    counter = Counter()
    limits = Limits(max_concurrency=2)

    await drain(limits, "stub", tasks=5, counter=counter)

    assert limits.in_flight("stub") == 0
    assert limits.peak("stub") == 2


def test_an_untouched_provider_reports_nothing() -> None:
    limits = Limits()

    assert (limits.in_flight("never-used"), limits.peak("never-used")) == (0, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_concurrency": 0},
        {"max_concurrency": -1},
        {"default_per_provider": 0},
        {"per_provider": {"gemini": 0}},
    ],
    ids=["global-zero", "global-negative", "default-zero", "provider-zero"],
)
def test_a_cap_below_one_is_rejected_rather_than_deadlocking(kwargs: dict[str, object]) -> None:
    # A cap of zero is a run that never starts. Fail loud at submit time (AGENTS.md §5).
    with pytest.raises(PolicyError, match="at least 1"):
        Limits(**kwargs)  # type: ignore[arg-type]
