"""A stateless worker: take a node off the queue, run it, report what happened.

This is the whole of the v2 execution tier, and its shortness is the point. It holds no
run state, no graph, and no schedule; everything it needs about a node it reads from the
``StateStore``, and everything it learns it writes back there. Start ten of them and they
are interchangeable — none is a leader, none has a queue of its own, and any of them can
finish a node another one started.

Three things it deliberately does *not* do.

It does not schedule. Which nodes are ready is a question about the graph, and the graph
has one owner; a worker that computed a ready set would be a second scheduler racing the
first.

It does not merge expansions. A planner running here can ask for nodes (FR-7), and the
request travels home on the result for the coordinator to validate — see
:class:`~dagent.runtime.node.CollectingExpansion`. The copy of the workflow a worker holds
may already be out of date, so it is not in a position to say yes.

It does not decide when it is done. The queue does, by holding a delivered message pending
until it is acknowledged. A worker killed between claiming a node and reporting it leaves
that message for somebody else, which is why :meth:`Worker.step` acknowledges *after*
publishing and never before.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

from dagent.errors import DagentError
from dagent.graph.topo import nodes_by_id
from dagent.models.state import NodeState
from dagent.models.workflow import Node
from dagent.observability import logging as obs_logging
from dagent.policy.run import RunPolicy
from dagent.runtime.clock import Clock
from dagent.runtime.model import ModelClient
from dagent.runtime.node import CollectingExpansion, NodeRunner, record_state
from dagent.runtime.registry import AgentRegistry
from dagent.store.base import StateStore
from dagent.transport.base import WorkItem, WorkQueue, WorkResult

__all__ = ["Worker"]

_CACHED_RUNS = 32
"""How many runs' definitions one worker keeps indexed.

A worker is long-lived and sees every run that passes through the pool, so an unbounded
cache is a slow leak. Small, because the cache exists to save a store read per node in one
run, not to remember runs that finished yesterday.
"""


@dataclass(frozen=True, slots=True)
class _Definition:
    """One run's stored definition, indexed for lookup."""

    name: str
    nodes: dict[str, Node]


class Worker:
    """Consumes nodes from a :class:`~dagent.transport.base.WorkQueue` and runs them."""

    def __init__(
        self,
        *,
        name: str,
        registry: AgentRegistry,
        store: StateStore,
        queue: WorkQueue,
        clock: Clock | None = None,
        model: ModelClient | None = None,
        policy: RunPolicy | None = None,
        claim_timeout_s: float = 1.0,
        reclaim_after_s: float = 30.0,
    ) -> None:
        """Wire the worker to its queue, its storage, and the agents it can run.

        Args:
            name: This worker's identity within the consumer group. Must be unique across
                the pool: two workers sharing a name are indistinguishable to
                :meth:`~dagent.transport.base.WorkQueue.reclaim`, so neither can tell that
                the other has died.
            registry: The agents this worker is able to execute. A worker that does not
                know a node's agent fails that node rather than guessing, which is why
                every worker in a pool should be running the same build.
            store: Where run state and outputs live. The same store the coordinator uses —
                it is the only thing the two processes share besides the queue.
            queue: The work transport.
            clock: The time seam, as everywhere else.
            model: The one provider client this worker shares across every node it runs.
            policy: Retries, timeouts, permits and budget. Enforced per worker, which for
                concurrency caps is what you want — each process gets its own slice of the
                provider's rate limit — and for a token *ceiling* is a real weakening, since
                nothing sums spending across processes.
            claim_timeout_s: How long one wait for new work blocks before the loop comes
                back round. Bounds how quickly the worker notices a cancellation, not how
                long a node may take.
            reclaim_after_s: How long a claimed-and-unacknowledged node must sit idle before
                another worker is entitled to take it over. This is the failure detector,
                and it is a timeout because there is no such thing as knowing a process is
                dead — only that it has not been heard from.
        """
        self._name = name
        self._queue = queue
        self._store = store
        self._claim_timeout_s = claim_timeout_s
        self._reclaim_after_s = reclaim_after_s
        self._runner = NodeRunner(
            registry=registry, store=store, clock=clock, model=model, policy=policy
        )
        self._definitions: dict[str, _Definition] = {}
        self._reclaimed: deque[WorkItem] = deque()

    async def run_forever(
        self, *, stop: asyncio.Event | None = None, limit: int | None = None
    ) -> int:
        """Consume work until asked to stop, and return how many nodes were run.

        Args:
            stop: Set it and the worker finishes the node it is on, then exits. A worker
                that abandoned its node mid-flight would be relying on redelivery to do
                what an orderly shutdown can do properly.
            limit: Stop after this many nodes. For scripted runs and tests; a real worker
                passes nothing and runs until it is told otherwise.

        Returns:
            The number of nodes this worker drove to a verdict.
        """
        log = obs_logging.get_logger()
        log.info("worker.started", worker=self._name)
        handled = 0
        while (stop is None or not stop.is_set()) and (limit is None or handled < limit):
            try:
                if await self.step() is not None:
                    handled += 1
            except DagentError as exc:
                # A store or transport failure is weather. The node stays unacknowledged
                # and somebody will pick it up; killing the worker over it would turn one
                # bad message into a lost process.
                log.warning("worker.step_failed", worker=self._name, error=str(exc))
        log.info("worker.stopped", worker=self._name, handled=handled)
        return handled

    async def step(self) -> WorkItem | None:
        """Take one node, run it, and report the result. Return the item, or ``None``.

        ``None`` means there was nothing to do within the claim timeout, which is the
        ordinary state of a worker pool that is bigger than its backlog.
        """
        item = await self._next()
        if item is None:
            return None

        result = await self._execute(item)
        # Publish, then acknowledge — in that order, and never the reverse. See
        # `WorkQueue.complete`: one order costs a duplicate execution under an idempotency
        # key that makes duplicates safe, the other costs a node nobody will ever run again.
        await self._queue.complete(item, result)
        return item

    async def _next(self) -> WorkItem | None:
        """Find the next node to run: reclaimed work first, then new work, then stealing.

        The order is what makes reclaiming cheap. A busy worker never asks whether anybody
        has died, because it has no spare capacity to do anything about it; the question is
        asked exactly when a claim came back empty, which is the moment this worker has
        room and the pool may be short a member.
        """
        if self._reclaimed:
            return self._reclaimed.popleft()

        item = await self._queue.claim(consumer=self._name, timeout_s=self._claim_timeout_s)
        if item is not None:
            return item

        self._reclaimed.extend(
            await self._queue.reclaim(consumer=self._name, min_idle_s=self._reclaim_after_s)
        )
        return self._reclaimed.popleft() if self._reclaimed else None

    async def _execute(self, item: WorkItem) -> WorkResult:
        """Run one claimed node and turn its outcome into a result to publish."""
        definition = await self._definition(item)
        node = definition.nodes.get(item.node_id)
        if node is None:
            # The coordinator asked for a node the stored definition does not contain.
            # Reported as a failure of that node rather than raised, because an
            # unacknowledged item would be handed to the next worker to fail identically.
            error = f"node {item.node_id!r} is not in the stored definition of run {item.run_id!r}"
            await record_state(
                self._store,
                item.run_id,
                item.node_id,
                NodeState.FAILED,
                attempt=item.attempt,
                error=f"ValidationError: {error}",
            )
            return WorkResult(item.run_id, item.node_id, item.attempt, NodeState.FAILED)

        expansion = CollectingExpansion()
        with obs_logging.bind_run(item.run_id, definition.name):
            outcome = await self._runner.execute(
                item.run_id, node, expand=expansion, first_attempt=item.attempt
            )
        return WorkResult(
            run_id=item.run_id,
            node_id=outcome.node_id,
            attempt=outcome.attempt,
            state=outcome.state,
            expansion=expansion.nodes,
        )

    async def _definition(self, item: WorkItem) -> _Definition:
        """Return the run's stored definition, reloading it if this node is not in it.

        Reload-on-miss is how a worker keeps up with a graph that grew: the coordinator
        persists an expansion before dispatching anything it added, so a node this worker
        has never heard of is always a signal that its copy is stale — never a signal that
        the node does not exist.
        """
        cached = self._definitions.pop(item.run_id, None)
        if cached is None or item.node_id not in cached.nodes:
            workflow = await self._store.load_workflow(item.run_id)
            cached = _Definition(workflow.name, nodes_by_id(workflow))
        # Reinserted rather than updated in place, so the dict's insertion order is a
        # genuine least-recently-used order and the eviction below drops the right run.
        self._definitions[item.run_id] = cached
        while len(self._definitions) > _CACHED_RUNS:
            del self._definitions[next(iter(self._definitions))]
        return cached
