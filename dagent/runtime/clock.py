"""The time seam.

Every read of wall-clock time and every sleep in dagent goes through a ``Clock``, so a
run can be replayed and a test can assert on timestamps without waiting for real seconds
(AGENTS.md rule 4, DR-5). ``SystemClock`` is the only place in the package that touches
real time.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Protocol

__all__ = ["Clock", "ManualClock", "SystemClock"]


class Clock(Protocol):
    """The injectable source of time."""

    def now(self) -> datetime:
        """Return the current time as a timezone-aware UTC datetime."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the caller for ``seconds`` without blocking the event loop."""
        ...


class SystemClock:
    """The real clock, backed by the wall clock and the event loop."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        """Sleep on the event loop."""
        await asyncio.sleep(seconds)


class ManualClock:
    """A clock a test drives by hand: it only moves when told to.

    Sleeping advances the clock instantly and records the requested duration, so a test
    can assert *that* something waited, and for how long, without waiting itself.
    """

    def __init__(self, start: datetime | None = None) -> None:
        """Start at ``start``, or at a fixed arbitrary instant if not given."""
        self._now = start if start is not None else datetime(2026, 1, 1, tzinfo=UTC)
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        """Return the time this clock has been advanced to."""
        return self._now

    def advance(self, seconds: float) -> None:
        """Move the clock forward."""
        self._now += timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        """Record the sleep and advance instantly, yielding once to the event loop."""
        self.sleeps.append(seconds)
        self.advance(seconds)
        # Still yield, so a caller that sleeps in a loop cannot starve its siblings.
        await asyncio.sleep(0)
