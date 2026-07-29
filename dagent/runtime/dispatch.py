"""Where a ready node goes to be run, and how its outcome comes back.

This is the one seam that separates single-process dagent from distributed dagent. The
executor's loop — compute the ready set, dispatch it, wait for the first completion, fold
the result in — is written against this protocol and does not know which implementation is
underneath. :class:`LocalDispatcher` runs nodes as ``asyncio`` tasks on the coordinator's
own loop, which is exactly what the executor did inline before Phase 8.
:class:`QueueDispatcher` posts them to a :class:`~dagent.transport.base.WorkQueue` and
waits for other processes to answer.

The interesting asymmetry is expansion. A node that grows the graph (FR-7) hands its
request to an :class:`~dagent.runtime.node.ExpansionSink`, and *who that sink is* is the
whole difference between the two transports:

* in one process, the sink is :class:`MergingExpansion` — it validates and merges on the
  spot, so the node is marked ``SUCCESS`` only after the grown graph is durable;
* across processes, the worker's sink merely collects, the request rides home on the
  result, and the coordinator merges it. It has to: the graph has exactly one owner, and a
  worker holding a copy of the workflow it loaded a second ago cannot know whether another
  worker's planner has grown it since.

Everything else — validation, the ready set, the failure modes, the retry policy, the
budget — is shared code that neither implementation touched. That is the claim DR-1 made
in Phase 0 and this module is where it is either true or it is not.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from dagent.models.state import NodeState
from dagent.models.workflow import Node
from dagent.runtime.expansion import RunGraph
from dagent.runtime.node import ExpansionSink, NodeOutcome, NodeRunner, record_state
from dagent.store.base import StateStore
from dagent.transport.base import WorkItem, WorkQueue, WorkResult

__all__ = ["Dispatcher", "LocalDispatcher", "MergingExpansion", "QueueDispatcher"]


@runtime_checkable
class Dispatcher(Protocol):
    """How the executor gets a node run and finds out what happened.

    One dispatcher serves one run at a time; :meth:`start` is what hands it the next one
    and resets whatever the last one left behind.
    """

    @property
    def outstanding(self) -> frozenset[str]:
        """Every node dispatched and not yet accounted for.

        The loop stops when this is empty and nothing is ready — "no further progress is
        possible", which is not the same as "no node is still ``PENDING``".
        """
        ...

    async def start(self, run_id: str, graph: RunGraph) -> Sequence[NodeOutcome]:
        """Begin serving a run, and report anything the last attempt at it left unfinished.

        Args:
            run_id: The run about to be driven.
            graph: The live graph. A dispatcher that runs nodes on this event loop uses it
                to merge expansions in place. A dispatcher that hands work to other
                processes ignores it, and *that* is the point: it cannot let them touch a
                graph they do not own, which is why expansion comes home in the result.

        Returns:
            Outcomes that were reported to a previous coordinator for this run and never
            acknowledged. Empty for a fresh run, and always empty for a dispatcher whose
            results never left the process — nothing in memory outlives the process that
            crashed.

            The caller must fold these in *before* deciding anything else. A resumed run
            has nothing outstanding by definition, so a coordinator that went straight to
            its loop would find nothing ready, nothing in flight, conclude that the run was
            over, and lose whatever those reports were carrying. That is not a hypothetical:
            a planner's expansion arrives this way, and losing it means a run that reports
            success having done none of the work it planned.
        """
        ...

    async def dispatch(self, node: Node, *, attempt: int) -> None:
        """Start one node at the given attempt number."""
        ...

    async def settle(self) -> Sequence[NodeOutcome]:
        """Wait for at least one outstanding node to finish, and report those that did."""
        ...

    async def acknowledge(self, outcomes: Sequence[NodeOutcome]) -> None:
        """Release outcomes the executor has finished folding in.

        Separate from :meth:`settle` so that a transport with durable delivery can hold a
        result until the coordinator has actually acted on it — merging an expansion,
        persisting the augmented graph — rather than forgetting it the instant it was read.
        A no-op for a dispatcher whose results never left the process.
        """
        ...

    async def halt(self) -> Sequence[NodeOutcome]:
        """Stop everything outstanding, and report what each node actually did.

        ``fail_fast`` calls this. What "stop" can mean depends on the transport, and the
        implementations differ honestly rather than pretending to match.
        """
        ...

    async def abandon(self) -> None:
        """Give up on everything outstanding without waiting for it.

        Used when the caller is leaving — a cancellation, or an error unwinding the loop.
        No outcomes, because there is no longer anybody to report them to.
        """
        ...


class MergingExpansion:
    """The expansion sink for a node running on the loop that owns the graph.

    Validates, merges, persists the augmented definition, and records the new nodes as
    ``PENDING`` — after which the ordinary ready-set loop picks them up with no special
    case, because a node missing from the state map already counted as ``PENDING``.
    """

    def __init__(self, run_id: str, graph: RunGraph, store: StateStore) -> None:
        """Bind the sink to one run's graph and its storage."""
        self._run_id = run_id
        self._graph = graph
        self._store = store

    async def absorb(self, node_id: str, nodes: Sequence[Node]) -> None:
        """Merge the request into the run graph and persist the result.

        ``RunGraph.apply`` is synchronous, so validate-and-merge cannot interleave with
        another node's expansion — the serialisation ARCHITECTURE §4 promises comes from
        the absence of an ``await``, not from a lock.

        Raises:
            ValidationError: If the expansion is rejected. It reaches the runner as an
                ordinary node failure, and the default retry classification does not retry
                it, so a planner that emits a cycle fails itself and nothing else.
        """
        added = self._graph.apply(node_id, nodes)
        if not added:
            return

        await self._store.save_workflow(self._run_id, self._graph.workflow)
        for node in added:
            await record_state(self._store, self._run_id, node.id, NodeState.PENDING)


class LocalDispatcher:
    """Runs nodes as tasks on the coordinator's own event loop — v1's transport.

    Kept as an implementation of the protocol rather than as a special case in the
    executor, so the single-process path is held to the same contract the distributed one
    is and cannot quietly acquire behaviour the other cannot have.
    """

    def __init__(self, runner: NodeRunner, *, store: StateStore) -> None:
        """Wire the dispatcher to the thing that runs a node and the thing that records it."""
        self._runner = runner
        self._store = store
        self._tasks: dict[asyncio.Task[NodeOutcome], tuple[str, int]] = {}
        self._run_id = ""
        self._expand: ExpansionSink | None = None

    @property
    def outstanding(self) -> frozenset[str]:
        """Every node whose task has not been collected."""
        return frozenset(node_id for node_id, _ in self._tasks.values())

    async def start(self, run_id: str, graph: RunGraph) -> Sequence[NodeOutcome]:
        """Serve this run, merging any expansion straight into its graph.

        Never reports anything unfinished: an in-process result is delivered by a
        coroutine returning, so there is no state in which one has been reported and not
        received. The durability that makes the queue's answer non-empty is exactly the
        durability an event loop does not have.
        """
        self._run_id = run_id
        self._expand = MergingExpansion(run_id, graph, self._store)
        self._tasks.clear()
        return ()

    async def dispatch(self, node: Node, *, attempt: int) -> None:
        """Spawn a task for one node."""
        assert self._expand is not None, "dispatch() before start()"
        task = asyncio.create_task(
            self._runner.execute(self._run_id, node, expand=self._expand, first_attempt=attempt),
            name=f"dagent:{self._run_id}:{node.id}",
        )
        self._tasks[task] = (node.id, attempt)

    async def settle(self) -> Sequence[NodeOutcome]:
        """Wait for the first task to finish, and take every task that finished with it."""
        done, _ = await asyncio.wait(set(self._tasks), return_when=asyncio.FIRST_COMPLETED)
        outcomes = []
        for task in done:
            self._tasks.pop(task)
            # Deliberately not swallowed: a task that raised is a bug in the engine rather
            # than a failing node, and it propagates to unwind the run loop.
            outcomes.append(task.result())
        return tuple(outcomes)

    async def acknowledge(self, outcomes: Sequence[NodeOutcome]) -> None:
        """Nothing to release: the result never left the process."""

    async def halt(self) -> Sequence[NodeOutcome]:
        """Cancel every running node and report what each of them actually did.

        A task whose agent swallowed the cancellation and returned anyway keeps its real
        outcome. Reporting a completed node as cancelled would be exactly the kind of lie
        the store must not contain.
        """
        await self._stop()
        outcomes = [
            NodeOutcome(node_id, attempt, NodeState.FAILED, cancelled=True)
            if task.cancelled()
            else task.result()
            for task, (node_id, attempt) in self._tasks.items()
        ]
        self._tasks.clear()
        return tuple(outcomes)

    async def abandon(self) -> None:
        """Cancel everything and wait for it to actually stop, discarding the outcomes."""
        await self._stop()
        self._tasks.clear()

    async def _stop(self) -> None:
        """Cancel every task and wait for it to finish stopping."""
        for task in self._tasks:
            task.cancel()
        # gather() of nothing is a no-op, so this needs no empty check.
        await asyncio.gather(*self._tasks, return_exceptions=True)


class QueueDispatcher:
    """Posts nodes to a work queue and waits for other processes to answer — v2.

    Holds no tasks and no agents. Everything it knows about a node in flight is that it
    was submitted, which is the honest limit of what a coordinator can know about work
    happening somewhere else.
    """

    def __init__(self, queue: WorkQueue, *, poll_timeout_s: float = 1.0) -> None:
        """Wire the dispatcher to its broker.

        ``poll_timeout_s`` bounds how long a single wait blocks. It is not a deadline on
        the work — a node may take as long as it likes — only on how often the coordinator
        gets its event loop back, which is what lets a cancellation be noticed promptly.
        """
        self._queue = queue
        self._poll_timeout_s = poll_timeout_s
        self._run_id = ""
        self._sent: dict[str, int] = {}
        self._unacknowledged: dict[str, WorkResult] = {}

    @property
    def outstanding(self) -> frozenset[str]:
        """Every node submitted and not yet reported back."""
        return frozenset(self._sent)

    async def start(self, run_id: str, graph: RunGraph) -> Sequence[NodeOutcome]:
        """Begin following this run's results channel, and take what was left unacknowledged.

        ``graph`` is ignored, and that is the design rather than an omission: the workers
        about to run these nodes are in other processes and must not be able to change the
        definition. Their expansion requests come back through :meth:`settle` instead, for
        the coordinator — the graph's one owner — to accept or reject.

        The unacknowledged reports are the other half of that bargain. Results are released
        only once they have been acted on, so whatever is still holding here is precisely
        the set a crashed coordinator read and never merged.
        """
        self._run_id = run_id
        self._sent.clear()
        self._unacknowledged.clear()
        # Before a single item is submitted: a results channel nobody is following yet is
        # one a fast worker can publish into and no one will ever read.
        await self._queue.follow(run_id)
        return self._receive(await self._queue.collect(run_id, timeout_s=0))

    async def dispatch(self, node: Node, *, attempt: int) -> None:
        """Offer one node to the pool of workers."""
        self._sent[node.id] = attempt
        await self._queue.submit(WorkItem(self._run_id, node.id, attempt))

    async def settle(self) -> Sequence[NodeOutcome]:
        """Block until at least one outstanding node reports back.

        Results for nodes this coordinator is not waiting for are acknowledged and dropped
        rather than folded in. At-least-once delivery makes duplicates ordinary, and a
        resumed run re-reads a channel that still holds completions from before the crash;
        neither is an error, and neither is news.
        """
        while True:
            results = await self._queue.collect(self._run_id, timeout_s=self._poll_timeout_s)
            stale = [result for result in results if result.node_id not in self._sent]
            if stale:
                await self._queue.settle(stale)

            outcomes = self._receive(result for result in results if result.node_id in self._sent)
            if outcomes:
                return outcomes

    async def acknowledge(self, outcomes: Sequence[NodeOutcome]) -> None:
        """Retire the results behind these outcomes, now that they have been acted on."""
        settled = [
            self._unacknowledged.pop(outcome.node_id)
            for outcome in outcomes
            if outcome.node_id in self._unacknowledged
        ]
        if settled:
            await self._queue.settle(settled)

    async def halt(self) -> Sequence[NodeOutcome]:
        """Stop dispatching, and wait for the nodes already out there to finish.

        A coroutine in another process cannot be cancelled — there is no handle to cancel
        with, and inventing one would mean a control channel every worker had to poll
        mid-node. So ``fail_fast`` degrades here from "cancel the siblings" to "start
        nothing further and let the siblings land", and the outcomes reported are the real
        ones. Recording a node as cancelled while a worker was still running it would put
        a falsehood in the store to preserve a word.
        """
        outcomes: list[NodeOutcome] = []
        while self._sent:
            outcomes.extend(await self.settle())
        await self.acknowledge(outcomes)
        return tuple(outcomes)

    async def abandon(self) -> None:
        """Walk away. Whatever is running keeps running; the store is where it lands."""
        self._sent.clear()
        self._unacknowledged.clear()

    def _receive(self, results: Iterable[WorkResult]) -> tuple[NodeOutcome, ...]:
        """Turn reports into outcomes, holding each one until it is acknowledged."""
        outcomes = []
        for result in results:
            self._sent.pop(result.node_id, None)
            self._unacknowledged[result.node_id] = result
            outcomes.append(
                NodeOutcome(
                    result.node_id, result.attempt, result.state, expansion=result.expansion
                )
            )
        return tuple(outcomes)
