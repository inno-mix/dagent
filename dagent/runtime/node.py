"""Running one node to a verdict — the unit of work, with no scheduler attached.

Everything a node needs in order to run correctly lives here: its attempts, its timeout,
the permits it holds, the budget it spends, the model calls it records, and the state it
writes. Nothing here knows what a *graph* is. It is handed one node and told which attempt
this is, and it returns what happened.

That separation is the whole of Phase 8. Until now this code was a private method on the
executor, which meant "run a node" was only reachable from "schedule a graph"; a second
process wanting to run a node had no way in that did not drag a scheduler with it. Splitting
it changes no behaviour — the retry loop, the ordering of the writes, and the metrics are
the ones Phases 4 through 7 built — but it is what lets :mod:`dagent.runtime.worker` exist
at all, and it is why the distributed executor needed no second copy of the policy layer.

The one genuinely new seam is :class:`ExpansionSink`. A node that grows the graph (FR-7)
cannot merge its own request, because the graph has exactly one owner and in a distributed
run that owner is in another process. So the runner hands the request to a sink and lets
the caller decide what owning it means.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from dagent.models.state import NodeOutput, NodeState, NodeStateRecord
from dagent.models.workflow import Node, Policy
from dagent.observability import logging as obs_logging
from dagent.observability import metrics as obs_metrics
from dagent.observability import tracing
from dagent.policy.run import RunPolicy
from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import Clock, SystemClock
from dagent.runtime.expansion import Expansion
from dagent.runtime.metering import BudgetedModelClient
from dagent.runtime.model import ModelClient, NullModelClient
from dagent.runtime.recording import RecordingModelClient
from dagent.runtime.registry import AgentRegistry
from dagent.store.base import StateStore

__all__ = [
    "CollectingExpansion",
    "ExpansionSink",
    "NodeOutcome",
    "NodeRunner",
    "record_state",
]


@dataclass(frozen=True, slots=True)
class NodeOutcome:
    """What happened to one node, as reported back to whoever is scheduling it.

    The runner fills in the first three fields and never the last two. ``expansion`` is
    filled in by a caller whose sink only *collected* the request rather than merging it,
    so that it can travel to whoever owns the graph; it stays empty in a single-process
    run, where the sink already merged it. ``cancelled`` is filled in by a dispatcher that
    stopped the node — a node cancelled by ``fail_fast`` did start and did not finish,
    which is a different thing from a node that reached a verdict of its own.
    """

    node_id: str
    attempt: int
    state: NodeState
    expansion: tuple[Node, ...] = ()
    cancelled: bool = False


class ExpansionSink(Protocol):
    """Where a node's request to grow the graph goes once the node has succeeded.

    The graph has one owner. In a single process that owner is the executor's
    :class:`~dagent.runtime.expansion.RunGraph`, reachable from the same event loop, so the
    sink merges the request on the spot and the node is only marked ``SUCCESS`` once the
    grown graph is durable. Across processes the owner is the coordinator, so the sink
    merely collects the request and the transport carries it home.

    One protocol, two answers to the same question: *who is allowed to change the graph?*
    """

    async def absorb(self, node_id: str, nodes: Sequence[Node]) -> None:
        """Take responsibility for the nodes ``node_id`` asked to add.

        Raises:
            ValidationError: If the request is rejected. It reaches the runner as an
                ordinary failure of ``node_id``, which is what stops a bad planner from
                deadlocking anything: it fails itself and the run carries on.
        """
        ...


@dataclass(slots=True)
class CollectingExpansion:
    """A sink for a runner that does not own the graph: hold the request, change nothing.

    Used by :mod:`dagent.runtime.worker`. Validation is deliberately *not* done here even
    though the worker has a copy of the workflow, because that copy may already be stale —
    another worker's planner may have grown the graph since it was loaded. Only the
    coordinator sees a graph nobody else is editing, so only the coordinator can say yes.
    """

    nodes: tuple[Node, ...] = ()

    async def absorb(self, node_id: str, nodes: Sequence[Node]) -> None:
        """Keep the request for the caller to ship back."""
        self.nodes = tuple(nodes)


async def record_state(
    store: StateStore,
    run_id: str,
    node_id: str,
    state: NodeState,
    *,
    attempt: int = 0,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    error: str | None = None,
) -> None:
    """Persist one node state transition.

    A free function rather than a method because both halves of a distributed run write
    these: the coordinator records ``READY`` and ``SKIPPED``, the worker records
    ``RUNNING`` and the verdict. One writer of the record shape means the two halves
    cannot drift into writing subtly different rows.
    """
    await store.save_node_state(
        NodeStateRecord(
            run_id=run_id,
            node_id=node_id,
            state=state,
            attempt=attempt,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
        )
    )


class NodeRunner:
    """Executes one node under its policy and persists what happened.

    Holds no run state and no graph: every call is self-contained, which is what makes it
    safe to share one runner across a worker's whole lifetime and across runs.
    """

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        store: StateStore,
        clock: Clock | None = None,
        model: ModelClient | None = None,
        policy: RunPolicy | None = None,
    ) -> None:
        """Wire the runner to its agents, storage, source of time, provider, and limits.

        The same five dependencies the executor takes, and deliberately so: a worker
        process is configured exactly like a single-process run, because it *is* one —
        minus the scheduler.
        """
        self._registry = registry
        self._store = store
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._model: ModelClient = model if model is not None else NullModelClient()
        self._policy = policy if policy is not None else RunPolicy()
        self._metrics = obs_metrics.metrics_for()

    async def execute(
        self,
        run_id: str,
        node: Node,
        *,
        expand: ExpansionSink,
        first_attempt: int = 0,
    ) -> NodeOutcome:
        """Run one node, retrying as its policy allows, and record every transition.

        Args:
            run_id: The run this node belongs to.
            node: The node to execute.
            expand: Who owns the graph, for a node that asks to grow it.
            first_attempt: Where to start counting. Non-zero only when a node is being
                re-dispatched after an interruption, where it must come back under the
                attempt — and therefore the idempotency key — it was already using.

        Returns:
            ``SUCCESS`` once an attempt produced an output, or ``FAILED`` when the attempts
            ran out or the failure was not worth repeating.
        """
        policy = self._policy.policy_for(node.policy)
        started = self._clock.now()
        attempt = first_attempt
        provider = self._model.provider
        labels = {obs_metrics.AGENT: node.agent, obs_metrics.PROVIDER: provider}
        log = obs_logging.get_logger()

        while True:
            # The whole attempt — dispatch, outcome, and the line that reports it — sits
            # inside one binding. The success log used to fall outside it and arrived with
            # no node_id, which is the one field FR-9 asks these lines to carry.
            with obs_logging.bind_node(node.id, node.agent, attempt):
                await record_state(
                    self._store,
                    run_id,
                    node.id,
                    NodeState.RUNNING,
                    attempt=attempt,
                    started_at=started,
                )
                if attempt > first_attempt:
                    self._metrics.retries.add(1, labels)
                try:
                    with tracing.node_span(run_id, node.id, node.agent, attempt) as span:
                        span.set_attribute(tracing.PROVIDER, provider)
                        self._metrics.nodes_in_flight.add(1, labels)
                        self._metrics.provider_in_flight.add(1, {obs_metrics.PROVIDER: provider})
                        try:
                            output = await self._attempt(
                                run_id,
                                node,
                                attempt=attempt,
                                policy=policy,
                                expand=expand,
                                span=span,
                            )
                        finally:
                            self._metrics.nodes_in_flight.add(-1, labels)
                            self._metrics.provider_in_flight.add(
                                -1, {obs_metrics.PROVIDER: provider}
                            )
                except Exception as exc:
                    # A failing agent is an ordinary outcome, recorded and folded back into
                    # the loop. CancelledError is not an Exception, so a cancelled node —
                    # and therefore a timed-out one, once its timeout has converted it —
                    # still propagates rather than being mistaken for a failure to retry.
                    log.warning("node.attempt_failed", error=f"{type(exc).__name__}: {exc}")
                    if attempt + 1 >= policy.max_attempts or not self._policy.retryable(exc):
                        self._metrics.nodes_completed.add(
                            1, {**labels, obs_metrics.STATE: NodeState.FAILED.value}
                        )
                        await record_state(
                            self._store,
                            run_id,
                            node.id,
                            NodeState.FAILED,
                            attempt=attempt,
                            started_at=started,
                            finished_at=self._clock.now(),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        return NodeOutcome(node.id, attempt, NodeState.FAILED)

                    # Deliberately no FAILED write between attempts. FAILED is terminal,
                    # and a terminal state a node then leaves is a lie — Phase 5's resume
                    # reads these records to decide what still needs doing. The rising
                    # `attempt` on the RUNNING record is the evidence that a retry happened.
                    await self._clock.sleep(self._policy.backoff.delay(policy, attempt))
                    attempt += 1
                    continue

                # Output first, then SUCCESS: a node found SUCCESS must always have an
                # output to read, or Phase 5's resume would skip a node whose result it
                # cannot recover. The expansion landed before both, inside `_attempt`, so a
                # graph that grew is durable before anything says the node that grew it is
                # done — a crash in between re-runs the planner, whose replayed expansion is
                # then a no-op.
                await self._store.append_output(run_id, node.id, output)
                await record_state(
                    self._store,
                    run_id,
                    node.id,
                    NodeState.SUCCESS,
                    attempt=attempt,
                    started_at=started,
                    finished_at=self._clock.now(),
                )
                self._metrics.nodes_completed.add(
                    1, {**labels, obs_metrics.STATE: NodeState.SUCCESS.value}
                )
                log.info("node.succeeded", attempt=attempt)
                return NodeOutcome(node.id, attempt, NodeState.SUCCESS)

    async def _attempt(
        self,
        run_id: str,
        node: Node,
        *,
        attempt: int,
        policy: Policy,
        expand: ExpansionSink,
        span: tracing.Span,
    ) -> NodeOutput:
        """Run the agent once, under this attempt's permits and timeout.

        A node that asked to grow the graph has that request handed on here, *after* it
        returned and before its output is persisted. Handing it on any earlier would let an
        attempt that expanded and then failed leave its nodes behind for the retry to trip
        over; any later would mark the node ``SUCCESS`` while the graph it promised to grow
        had not grown.
        """
        agent = self._registry.create(node.agent)
        inputs = await self._resolve_inputs(run_id, node)
        expansion = Expansion()
        before = (self._policy.budget.tokens_used, self._policy.budget.cost_used)
        context = AgentContext(
            run_id=run_id,
            node_id=node.id,
            attempt=attempt,
            inputs=inputs,
            params=node.params,
            clock=self._clock,
            expansion=expansion,
            # Budget outermost, so a refused call is never recorded: there is no response
            # to replay. Inside it, a per-node recorder over the run's shared client, so
            # every call this attempt makes lands in the run record for replay (FR-8).
            model=BudgetedModelClient(
                RecordingModelClient(
                    self._model,
                    store=self._store,
                    run_id=run_id,
                    node_id=node.id,
                    attempt=attempt,
                    clock=self._clock,
                ),
                budget=self._policy.budget,
                price=self._policy.price,
            ),
        )

        # Permits are acquired per attempt, not per node: holding a slot through a backoff
        # sleep would block a node that could be running. The timeout starts inside them,
        # because queueing for a permit is contention rather than the node's own latency,
        # and a node that never got to run has not overrun anything.
        async with (
            self._policy.limits.slot(self._model.provider),
            asyncio.timeout(policy.timeout_s),
        ):
            output = await agent.run(context)

        if expansion:
            await expand.absorb(node.id, expansion.nodes)
        self._record_usage(span, before)
        return output

    def _record_usage(self, span: tracing.Span, before: tuple[int, float]) -> None:
        """Attribute this attempt's model spend to its span and to the counters.

        Measured as the budget's movement rather than by inspecting the response, because
        the budget is the one place every call is already metered — and a node that made
        three calls should report the total, not the last one.
        """
        tokens = self._policy.budget.tokens_used - before[0]
        cost = self._policy.budget.cost_used - before[1]
        if not tokens and not cost:
            return

        provider = {obs_metrics.PROVIDER: self._model.provider}
        span.set_attribute(tracing.OUTPUT_TOKENS, tokens)
        self._metrics.tokens.add(tokens, provider)
        if cost:
            self._metrics.cost.add(cost, provider)

    async def _resolve_inputs(self, run_id: str, node: Node) -> Mapping[str, NodeOutput]:
        """Read this node's declared inputs out of its upstream nodes' stored outputs.

        Reading through the store rather than from an in-process cache is what makes the
        store the source of truth — and it is what lets a node run in a process that never
        saw the node that produced its input. Validation guarantees each source is an
        ancestor, so by the time this runs the output is there.
        """
        return {
            local_name: await self._store.load_output(run_id, source)
            for local_name, source in node.inputs.items()
        }
