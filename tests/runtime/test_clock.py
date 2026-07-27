from datetime import UTC, datetime

import pytest

from dagent.runtime.clock import Clock, ManualClock, SystemClock


def test_system_clock_satisfies_the_protocol() -> None:
    clock: Clock = SystemClock()
    assert clock is not None


def test_manual_clock_satisfies_the_protocol() -> None:
    clock: Clock = ManualClock()
    assert clock is not None


def test_system_clock_returns_timezone_aware_utc() -> None:
    # A naive datetime would compare wrongly against anything the store round-trips.
    assert SystemClock().now().tzinfo is not None


def test_manual_clock_does_not_move_on_its_own() -> None:
    clock = ManualClock()

    assert clock.now() == clock.now()


def test_manual_clock_starts_where_it_is_told() -> None:
    start = datetime(2030, 6, 1, tzinfo=UTC)

    assert ManualClock(start).now() == start


def test_advancing_moves_the_manual_clock_forward() -> None:
    clock = ManualClock(datetime(2030, 1, 1, tzinfo=UTC))

    clock.advance(90)

    assert clock.now() == datetime(2030, 1, 1, 0, 1, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_manual_clock_sleeps_instantly_and_records_the_duration() -> None:
    # This is the point of the seam: Phase 4 can assert on backoff without waiting for it.
    clock = ManualClock(datetime(2030, 1, 1, tzinfo=UTC))

    await clock.sleep(30)
    await clock.sleep(12)

    assert clock.sleeps == [30, 12]
    assert clock.now() == datetime(2030, 1, 1, 0, 0, 42, tzinfo=UTC)


@pytest.mark.asyncio
async def test_system_clock_sleep_yields_to_the_event_loop() -> None:
    before = SystemClock().now()

    await SystemClock().sleep(0.001)

    assert SystemClock().now() >= before
