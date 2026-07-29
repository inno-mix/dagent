"""An in-process ``WorkQueue``: consumer-group semantics with no broker behind them.

The point of this implementation is not speed, and it is not to be a fallback. It is that
the distributed executor, the worker loop, and every property Phase 8 claims — redelivery
after a crash, at-least-once safety, two workers never running the same item at once — can
be *tested* with no services running at all. A test suite that needs a broker is a test
suite that gets skipped, and a skipped test proves nothing.

So the pending-entry bookkeeping here is real rather than approximated: a claimed item is
held until it is acknowledged, an unacknowledged item goes stale and can be reclaimed by
somebody else, and a reclaimed item keeps its original receipt exactly as Redis'
``XAUTOCLAIM`` does. Where this differs from Redis, the conformance suite is what finds
out.

Time comes from the event loop's own clock (``loop.time()``) rather than from the injected
``Clock``: idle time here is a fact about scheduling, in the same way a timeout is (DR-10),
and ``dagent.transport`` sits below ``dagent.runtime``, where the ``Clock`` lives.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace

from dagent.transport.base import WorkItem, WorkResult

__all__ = ["InMemoryWorkQueue"]


@dataclass(slots=True)
class _Delivery:
    """One item handed to one consumer, and not yet acknowledged."""

    item: WorkItem
    consumer: str
    delivered_at: float


class InMemoryWorkQueue:
    """A single-process broker: one work channel, one results channel per run."""

    def __init__(self) -> None:
        """Start empty, with nothing followed and nothing pending."""
        self._backlog: deque[WorkItem] = deque()
        self._pending: dict[str, _Delivery] = {}
        self._work_ready = asyncio.Event()

        self._results: dict[str, deque[WorkResult]] = {}
        self._results_ready: dict[str, asyncio.Event] = {}
        self._unsettled: dict[str, WorkResult] = {}

        self._receipts = itertools.count(1)

    # --- the coordinator's half ---------------------------------------------------------

    async def follow(self, run_id: str) -> None:
        """Make sure this run has somewhere for its results to land."""
        self._results.setdefault(run_id, deque())
        self._results_ready.setdefault(run_id, asyncio.Event())

    async def submit(self, item: WorkItem) -> None:
        """Add an item to the backlog and wake a worker."""
        self._backlog.append(item)
        self._work_ready.set()

    async def collect(self, run_id: str, *, timeout_s: float = 1.0) -> Sequence[WorkResult]:
        """Return this run's waiting results, unacknowledged ones first."""
        unsettled = tuple(result for result in self._unsettled.values() if result.run_id == run_id)
        if unsettled:
            return unsettled

        await self.follow(run_id)
        queue = self._results[run_id]
        ready = self._results_ready[run_id]
        if not queue and not await _wait(ready, timeout_s):
            return ()

        delivered = []
        while queue:
            stamped = replace(queue.popleft(), receipt=self._receipt())
            self._unsettled[stamped.receipt] = stamped
            delivered.append(stamped)
        ready.clear()
        return tuple(delivered)

    async def settle(self, results: Sequence[WorkResult]) -> None:
        """Forget results the coordinator has finished acting on."""
        for result in results:
            self._unsettled.pop(result.receipt, None)

    # --- a worker's half ----------------------------------------------------------------

    async def claim(self, *, consumer: str, timeout_s: float = 1.0) -> WorkItem | None:
        """Take the next item, waiting up to ``timeout_s`` for one to appear."""
        deadline = _now() + timeout_s
        while True:
            if self._backlog:
                item = replace(self._backlog.popleft(), receipt=self._receipt())
                if not self._backlog:
                    self._work_ready.clear()
                self._pending[item.receipt] = _Delivery(item, consumer, _now())
                return item
            # Looping rather than returning on the first wake-up: several workers can be
            # woken by one submission, and the ones that lose the race are still inside
            # the window their caller asked to wait for.
            if not await _wait(self._work_ready, deadline - _now()):
                return None

    async def reclaim(self, *, consumer: str, min_idle_s: float) -> Sequence[WorkItem]:
        """Take over every delivery that has been idle too long.

        The receipt is preserved, as ``XAUTOCLAIM`` preserves a message id: the delivery is
        being *transferred*, not reissued, so an acknowledgement from whichever worker
        finishes first retires the same entry.
        """
        now = _now()
        stale = [
            delivery
            for delivery in self._pending.values()
            if now - delivery.delivered_at >= min_idle_s
        ]
        for delivery in stale:
            delivery.consumer = consumer
            delivery.delivered_at = now
        return tuple(delivery.item for delivery in stale)

    async def complete(self, item: WorkItem, result: WorkResult) -> None:
        """Publish the result, then retire the item that produced it."""
        await self.follow(result.run_id)
        self._results[result.run_id].append(result)
        self._results_ready[result.run_id].set()
        self._pending.pop(item.receipt, None)

    async def close(self) -> None:
        """Nothing to release."""

    # --- inspection, for tests ----------------------------------------------------------

    @property
    def pending(self) -> Sequence[WorkItem]:
        """Every item delivered to a worker and not yet acknowledged."""
        return tuple(delivery.item for delivery in self._pending.values())

    @property
    def backlog(self) -> Sequence[WorkItem]:
        """Every item submitted and not yet delivered to anyone."""
        return tuple(self._backlog)

    def _receipt(self) -> str:
        """Mint a handle for one delivery."""
        return str(next(self._receipts))


def _now() -> float:
    """The event loop's clock, which is the one idle time is measured against."""
    return asyncio.get_running_loop().time()


async def _wait(event: asyncio.Event, timeout_s: float) -> bool:
    """Wait for an event, returning whether it fired before the deadline."""
    if timeout_s <= 0:
        return event.is_set()
    try:
        await asyncio.wait_for(event.wait(), timeout_s)
    except TimeoutError:
        return False
    return True
