"""The ``WorkQueue`` protocol: how a ready node reaches the process that will run it.

This is the seam DR-1 promised. v1's transport was ``asyncio.create_task`` — a queue with
no queue, where dispatch and execution happen on the same event loop. v2 puts a broker in
the middle, and the only thing that had to change to allow it was the shape of this
protocol, because the scheduler was never coupled to how a node gets from ready to running.

Two channels, in opposite directions, and they are deliberately not symmetric:

* **work**, one stream shared by every run, consumed by a group of interchangeable
  workers. Any worker can run any node, which is what makes workers stateless and
  horizontally scalable.
* **results**, one stream *per run*, consumed only by that run's coordinator. There is
  exactly one coordinator per run — it is the thing that owns the graph — so a shared
  results channel would mean every coordinator filtering out everyone else's traffic.

Both are at-least-once. A message is acknowledged only once the work it describes has
been *dealt with*, never on receipt, so a consumer that dies mid-message leaves it pending
for someone else to pick up. That is the guarantee Phase 5's idempotency key was built
for: the redelivered item names the same ``(run_id, node_id, attempt)``, so the node comes
back under the key the outside world has already seen and an agent that deduplicates on
it commits exactly once (DR-4).

Implementations: :mod:`dagent.transport.memory` (asyncio, no services, used by the tests)
and :mod:`dagent.transport.redis` (Redis Streams and consumer groups). Both are held to
one contract by ``tests/transport/test_conformance.py``, because "the transport is
swappable" is only a claim worth making if two of them actually agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dagent.models.state import NodeState
from dagent.models.workflow import Node

__all__ = ["WorkItem", "WorkQueue", "WorkResult"]


@dataclass(frozen=True, slots=True)
class WorkItem:
    """One node, handed to whichever worker takes it next.

    The payload is the idempotency key and nothing else. Not the node definition, not its
    inputs, not the workflow: all of that is already in the ``StateStore``, and a worker
    that reads it from there cannot be looking at a stale copy of a graph that has since
    grown. A message that carried the definition would be a second source of truth, and
    the two would disagree the first time a planner expanded anything.
    """

    run_id: str
    node_id: str
    attempt: int
    receipt: str = ""
    """The transport's handle on *this delivery*, for acknowledging it.

    Empty on an item being submitted; filled in by the queue on the way out. Two
    deliveries of the same item — which at-least-once makes possible — carry the same
    ``(run_id, node_id, attempt)`` and different receipts, which is the difference between
    identifying the work and identifying the attempt to hand it over.
    """

    @property
    def idempotency_key(self) -> str:
        """``(run_id, node_id, attempt)`` as a string — the same one the agent sees."""
        return f"{self.run_id}:{self.node_id}:{self.attempt}"


@dataclass(frozen=True, slots=True)
class WorkResult:
    """What a worker reports back once it has driven a node to a verdict.

    Carries the expansion a planner asked for, because the worker is not allowed to merge
    it: the graph has one owner and it is the coordinator. See
    :class:`~dagent.runtime.node.ExpansionSink`.
    """

    run_id: str
    node_id: str
    attempt: int
    state: NodeState
    expansion: tuple[Node, ...] = ()
    receipt: str = ""
    """The results channel's handle on this delivery, filled in by the queue."""


@runtime_checkable
class WorkQueue(Protocol):
    """A broker between one coordinator and many workers.

    Split by role rather than by verb: :meth:`submit`, :meth:`follow`, :meth:`collect` and
    :meth:`settle` are the coordinator's half, :meth:`claim`, :meth:`reclaim` and
    :meth:`complete` are a worker's. Nothing enforces that division — it is a protocol,
    not a permission system — but keeping it visible is what makes the two processes easy
    to reason about separately.
    """

    async def follow(self, run_id: str) -> None:
        """Start retaining results for a run, before any of its work is submitted.

        Called by the coordinator when a run opens. It has to happen first: a results
        channel nobody is following yet is one a worker can publish into and no one will
        ever read, and the node whose completion was dropped would hang the run.

        Safe to call again for a run already being followed, which is what makes ``resume``
        work — the second call picks up whatever the first coordinator never acknowledged.
        """
        ...

    async def submit(self, item: WorkItem) -> None:
        """Offer a node to the pool of workers."""
        ...

    async def collect(self, run_id: str, *, timeout_s: float = 1.0) -> Sequence[WorkResult]:
        """Return whatever results are waiting for a run, blocking up to ``timeout_s``.

        Returns an empty sequence on timeout rather than raising: a coordinator with
        nothing to do is the normal case, not an error, and it wants the loop back so it
        can notice a cancellation.

        Results already delivered but never acknowledged come back first. That is what
        closes the window a coordinator crash opens: a run whose planner succeeded but
        whose expansion was never merged gets the report again rather than losing the
        nodes it described.
        """
        ...

    async def settle(self, results: Sequence[WorkResult]) -> None:
        """Acknowledge results the coordinator has finished acting on.

        Deliberately not folded into :meth:`collect`. Acknowledging on receipt would mean
        a result was forgotten the instant it was read, and everything the coordinator does
        with one — merging an expansion, persisting the augmented graph — happens *after*
        it is read. Acknowledging afterwards is what makes those steps re-runnable.
        """
        ...

    async def claim(self, *, consumer: str, timeout_s: float = 1.0) -> WorkItem | None:
        """Take the next item of work, blocking up to ``timeout_s``.

        Returns ``None`` on timeout. ``consumer`` names this worker within the group, and
        is what lets :meth:`reclaim` tell a worker that is merely slow from one that died.
        """
        ...

    async def reclaim(self, *, consumer: str, min_idle_s: float) -> Sequence[WorkItem]:
        """Take over work that was claimed and never acknowledged.

        This is how a dead worker's node gets run at all. Consumer groups hold a delivered
        message as *pending* until it is acknowledged, so a worker killed mid-node leaves
        an entry that no amount of waiting will clear on its own; another worker has to
        notice it has gone stale and claim it.

        ``min_idle_s`` is the only defence against stealing work from a worker that is
        simply slow, and stealing it is not a correctness failure — the node just runs
        twice under one idempotency key, which is exactly the case Phase 5 made safe.
        """
        ...

    async def complete(self, item: WorkItem, result: WorkResult) -> None:
        """Publish a result, then acknowledge the item that produced it.

        In that order, and never the reverse. Acknowledging first would let a crash in
        between erase the work from the queue while its result was never reported, and the
        run would wait for a node nobody is going to run again. Publishing first risks the
        opposite — the result is reported and the item is redelivered — which costs one
        duplicate execution under the same idempotency key and costs nothing else.
        """
        ...

    async def close(self) -> None:
        """Release whatever the queue is holding open."""
        ...
