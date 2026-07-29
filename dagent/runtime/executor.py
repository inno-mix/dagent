"""The scheduler: the ready-set run loop that drives a workflow to completion.

The whole engine is one loop — compute the ready set, dispatch it, wait for the first
completion, fold the result in, repeat. Fan-out is dispatching several ready nodes at
once; fan-in is the recomputation afterwards noticing that a node's last dependency just
landed. Nothing here knows what an agent does, and nothing here imports a model SDK.

Policy wraps that loop rather than complicating it. A node's attempts, its timeout, the
permits it holds and the budget it spends are all decided in ``dagent.policy`` and merely
*applied* here, which is why adding retries did not change the loop by a line.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from dagent.errors import StoreError, ValidationError
from dagent.graph.topo import descendants, nodes_by_id, ready_set
from dagent.graph.validate import validate
from dagent.models.state import (
    NodeOutput,
    NodeState,
    NodeStateRecord,
    RunState,
    RunStateRecord,
)
from dagent.models.workflow import Node, Policy, Workflow
from dagent.observability import logging as obs_logging
from dagent.observability import metrics as obs_metrics
from dagent.observability import tracing
from dagent.policy.run import FailureMode, RunPolicy
from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import Clock, SystemClock
from dagent.runtime.expansion import Expansion, RunGraph
from dagent.runtime.metering import BudgetedModelClient
from dagent.runtime.model import ModelClient, NullModelClient
from dagent.runtime.recording import RecordingModelClient
from dagent.runtime.registry import AgentRegistry
from dagent.store.base import StateStore

__all__ = ["Executor"]

_INTERRUPTED = frozenset({NodeState.PENDING, NodeState.READY, NodeState.RUNNING})
"""Node states a resumed run re-dispatches.

``PENDING`` never started. ``READY`` was dispatched but may not have reached its agent.
``RUNNING`` was mid-flight when the process died, and nothing on this side of the crash
can tell you how far it got — which is exactly why it comes back under the same
idempotency key rather than a fresh one.
"""


class Executor:
    """Runs a validated workflow to completion on a single event loop.

    Everything the executor depends on is injected: the agents come from a registry, the
    durability from a ``StateStore`` protocol, time from a ``Clock``, and every limit from
    a ``RunPolicy``. That is what keeps this class testable with no network, no database,
    and no waiting.
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
        """Wire the executor to its agents, storage, source of time, provider, and limits.

        One ``model`` client is shared by the whole run (AGENTS.md §4: never a client per
        call). Each node gets thin recording and metering wrappers around it, not a
        connection of its own. Omit it and agents that try to call a model fail with a
        clear message.

        Omitting ``policy`` gives a run with no caps, no ceiling, one attempt per node and
        no timeout — the behaviour of an executor with no policy layer at all, so a limit
        is always something a caller asked for.
        """
        self._registry = registry
        self._store = store
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._model: ModelClient = model if model is not None else NullModelClient()
        self._policy = policy if policy is not None else RunPolicy()
        # Built once: instrument creation is not free, and the loop records on every
        # node transition. Observability never feeds back into a run's outcome, which
        # is why this one dependency is resolved rather than injected — see DR-11.
        self._metrics = obs_metrics.metrics_for()

    async def run(self, workflow: Workflow, *, run_id: str) -> RunStateRecord:
        """Execute every node in the workflow and return the final run state.

        ``run_id`` is supplied rather than generated: an id minted in here would be one
        more piece of non-determinism the caller cannot reproduce (AGENTS.md rule 4).

        Args:
            workflow: The definition to run. Validated first, and never mutated.
            run_id: Identifies this run in the store.

        Returns:
            The run's final state, read back from the store.

        Raises:
            ValidationError: If the workflow is invalid, names an unregistered agent, or
                ``run_id`` is already taken. Nothing executes and nothing is written.
            CancelledError: If the caller cancels the run. Every node in flight is stopped
                first and the run is checkpointed ``CANCELLED`` before this propagates.
        """
        # Submit-time validation, with the registry injected — this is the caller that
        # actually has one, which is why graph/ takes it as a parameter.
        validate(workflow, known_agents=self._registry.names())
        await self._require_unused(run_id)

        states: dict[str, NodeState] = dict.fromkeys(nodes_by_id(workflow), NodeState.PENDING)
        await self._open_run(run_id, workflow, states)
        return await self._drive(run_id, self._live_graph(workflow), states, attempts={})

    async def resume(self, run_id: str) -> RunStateRecord:
        """Continue a run that was interrupted, and return its final state.

        Takes no workflow: the definition was persisted with the run, so resuming cannot
        pick up a different graph than the one that was interrupted — and a Phase 6 graph
        that a planner grew at run time is resumable even though no file describes it.

        What each node does depends only on the state it was left in:

        * ``SUCCESS`` — skipped. Its output is already in the store, so its dependents can
          read it without the node running again.
        * ``RUNNING`` or ``READY`` — interrupted, and re-dispatched **at the same attempt
          number**. A crash cannot tell you whether the side effect landed, so the node
          re-runs under the same idempotency key the outside world already saw.
        * ``FAILED`` or ``SKIPPED`` — a decision was reached; resume does not revisit it.
        * ``PENDING`` — never started, and picked up normally.

        Budget usage is rebuilt from the recorded model calls before anything runs. A
        per-run ceiling that reset itself every time the process died would not be a
        per-run ceiling.

        Raises:
            StoreError: If the run or its definition is not in the store.
            ValidationError: If the run already succeeded — there is nothing to continue —
                or if the stored definition no longer validates, which is what a since
                removed agent looks like.
        """
        run = await self._store.load_run(run_id)
        if run.state is RunState.SUCCEEDED:
            raise ValidationError(f"run {run_id!r} already succeeded; there is nothing to resume")

        workflow = await self._store.load_workflow(run_id)
        validate(workflow, known_agents=self._registry.names())

        states: dict[str, NodeState] = {}
        attempts: dict[str, int] = {}
        for node_id in nodes_by_id(workflow):
            record = run.nodes.get(node_id)
            if record is None or record.state in _INTERRUPTED:
                # Back to PENDING so ready_set offers it again — carrying the attempt it
                # was on, which is what keeps the idempotency key stable across the crash.
                states[node_id] = NodeState.PENDING
                attempts[node_id] = record.attempt if record is not None else 0
                continue
            states[node_id] = record.state

        await self._rehydrate_budget(run_id)
        await self._checkpoint(run_id, RunState.RUNNING)
        return await self._drive(run_id, self._live_graph(workflow), states, attempts)

    def _live_graph(self, workflow: Workflow) -> RunGraph:
        """Wrap a definition in the holder that lets it grow (FR-7).

        A resumed run treats every node in the reloaded graph as generation zero, so the
        depth limit bounds one run *attempt* rather than a run across many crashes. The
        bound that survives a crash is ``max_graph_nodes``, which is measured against the
        graph as persisted. Recording provenance per node would close the gap, at the cost
        of putting an engine concept into the user-facing schema; the node ceiling buys
        the same safety for nothing.
        """
        return RunGraph(
            workflow,
            known_agents=self._registry.names(),
            max_depth=self._policy.max_expansion_depth,
            max_nodes=self._policy.max_graph_nodes,
        )

    async def _drive(
        self,
        run_id: str,
        graph: RunGraph,
        states: dict[str, NodeState],
        attempts: Mapping[str, int],
    ) -> RunStateRecord:
        """The run loop, shared by a fresh run and a resumed one.

        There is exactly one of these on purpose: a resume that went down a second code
        path would be a second set of scheduling bugs, and the whole claim of DR-4 is that
        resume is *not* special — it is the same loop over reloaded state.

        The graph is re-read from ``graph`` on every pass rather than captured once,
        because a node that expanded it since the last pass has already changed it. Nodes
        that appeared this way are simply absent from ``states``, and ``ready_set`` counts
        an absent node as ``PENDING`` — Phase 1 chose that rule for exactly this moment, so
        expansion needs no state backfill and no second scheduling path.
        """
        in_flight: dict[asyncio.Task[NodeState], str] = {}
        started = self._clock.now()
        log = obs_logging.get_logger()
        try:
            with (
                obs_logging.bind_run(run_id, graph.workflow.name),
                tracing.run_span(run_id, graph.workflow.name) as span,
            ):
                log.info("run.started", nodes=len(graph.workflow.nodes))
                state = await self._loop(run_id, graph, states, attempts, in_flight, log)
                span.set_attribute(tracing.RUN_STATE, state.value)
        except asyncio.CancelledError:
            # The caller walked away: stop every node, then say so in the record rather
            # than leaving a run that looks like it is still going.
            await self._cancel(in_flight)
            await self._checkpoint(run_id, RunState.CANCELLED)
            raise
        except BaseException:
            await self._cancel(in_flight)
            raise

        self._metrics.run_duration.record(
            (self._clock.now() - started).total_seconds(),
            {obs_metrics.WORKFLOW: graph.workflow.name, obs_metrics.STATE: state.value},
        )
        log.info("run.finished", state=state.value, nodes=len(graph.workflow.nodes))
        return await self._checkpoint(run_id, state)

    async def _loop(
        self,
        run_id: str,
        graph: RunGraph,
        states: dict[str, NodeState],
        attempts: Mapping[str, int],
        in_flight: dict[asyncio.Task[NodeState], str],
        log: Any,
    ) -> RunState:
        """Drive the ready set until no further progress is possible.

        Split out from :meth:`_drive` so the run span and the log binding wrap it as one
        block rather than being threaded through the loop body. Cancellation handling
        stays outside, where it can still checkpoint after the context managers unwind.
        """
        halted = False
        while True:
            workflow = graph.workflow
            nodes = nodes_by_id(workflow)
            # Declaration order, so a batch of completions is folded in the same
            # sequence every time — see the comment where `declared` is used below.
            declared = {node_id: index for index, node_id in enumerate(nodes)}

            ready = () if halted else ready_set(workflow, states)
            self._metrics.ready_set_size.record(len(ready), {obs_metrics.WORKFLOW: workflow.name})
            for node_id in ready:
                # Mark READY before dispatching: ready_set only offers PENDING
                # nodes, so this is what stops the next pass re-dispatching one.
                states[node_id] = NodeState.READY
                await self._record(
                    run_id, node_id, NodeState.READY, attempt=attempts.get(node_id, 0)
                )
                task = asyncio.create_task(
                    self._execute(
                        run_id,
                        nodes[node_id],
                        graph=graph,
                        first_attempt=attempts.get(node_id, 0),
                    ),
                    name=f"dagent:{run_id}:{node_id}",
                )
                in_flight[task] = node_id

            # Nothing running and nothing ready: the graph can make no more progress.
            # With a failed node that leaves its dependents PENDING — see _close_run.
            if not in_flight:
                break

            done, _ = await asyncio.wait(set(in_flight), return_when=asyncio.FIRST_COMPLETED)
            # `asyncio.wait` hands back a *set*, and several nodes can land in the
            # same batch. Folding them in set order would make the record depend on
            # hash iteration — two identical runs could then attribute a skipped node
            # to different upstream failures. Declaration order restores rule 4.
            failures: list[str] = []
            for task in sorted(done, key=lambda finished: declared[in_flight[finished]]):
                node_id = in_flight.pop(task)
                states[node_id] = task.result()
                if states[node_id] is NodeState.FAILED:
                    failures.append(node_id)

            for node_id in failures:
                log.warning("node.failed", node_id=node_id)
                halted |= await self._apply_failure_mode(
                    run_id, graph.workflow, node_id, states, in_flight
                )

        return self._derive_run_state(graph.workflow, states)

    # --- one node -------------------------------------------------------------------

    async def _execute(
        self, run_id: str, node: Node, *, graph: RunGraph, first_attempt: int = 0
    ) -> NodeState:
        """Run one node under its own task, retrying as its policy allows.

        Args:
            run_id: The run this node belongs to.
            node: The node to execute.
            graph: The live graph, so a node that asks to expand it can be answered.
            first_attempt: Where to start counting. Non-zero only on resume, where an
                interrupted node re-runs under the attempt — and therefore the
                idempotency key — it was already using when the process died.

        Returns:
            ``SUCCESS`` once an attempt produced an output, or ``FAILED`` when the
            attempts ran out or the failure was not worth repeating.
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
                await self._record(
                    run_id, node.id, NodeState.RUNNING, attempt=attempt, started_at=started
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
                                graph=graph,
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
                        await self._record(
                            run_id,
                            node.id,
                            NodeState.FAILED,
                            attempt=attempt,
                            started_at=started,
                            finished_at=self._clock.now(),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        return NodeState.FAILED

                    # Deliberately no FAILED write between attempts. FAILED is terminal,
                    # and a terminal state a node then leaves is a lie — Phase 5's resume
                    # reads these records to decide what still needs doing. The rising
                    # `attempt` on the RUNNING record is the evidence that a retry happened.
                    await self._clock.sleep(self._policy.backoff.delay(policy, attempt))
                    attempt += 1
                    continue

                # Output first, then SUCCESS: a node found SUCCESS must always have an
                # output to read, or Phase 5's resume would skip a node whose result it
                # cannot recover. The expansion lands before both, so a graph that grew is
                # durable before anything says the node that grew it is done — a crash in
                # between re-runs the planner, whose replayed expansion is then a no-op.
                await self._store.append_output(run_id, node.id, output)
                await self._record(
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
                return NodeState.SUCCESS

    async def _attempt(
        self,
        run_id: str,
        node: Node,
        *,
        attempt: int,
        policy: Policy,
        graph: RunGraph,
        span: tracing.Span,
    ) -> NodeOutput:
        """Run the agent once, under this attempt's permits and timeout.

        A node that asked to grow the graph has that request applied here, *after* it
        returned and before its output is persisted. Applying it any earlier would let an
        attempt that expanded and then failed leave its nodes behind for the retry to trip
        over; applying it any later would mark the node ``SUCCESS`` while the graph it
        promised to grow had not grown.
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

        await self._apply_expansion(run_id, node.id, graph, expansion)
        self._record_usage(span, before)
        return output

    async def _apply_expansion(
        self, run_id: str, node_id: str, graph: RunGraph, expansion: Expansion
    ) -> None:
        """Merge a node's requested expansion into the run, and persist the result.

        ``graph.apply`` is synchronous, so validate-and-merge cannot interleave with
        another node's expansion — the serialisation ARCHITECTURE §4 promises comes from
        the absence of an ``await``, not from a lock.

        Raises:
            ValidationError: If the expansion is rejected. It reaches the caller as an
                ordinary node failure, and the default retry classification does not retry
                it, so a planner that emits a cycle fails itself and nothing else. The run
                carries on and terminates normally rather than deadlocking.
        """
        added = graph.apply(node_id, expansion)
        if not added:
            return

        await self._store.save_workflow(run_id, graph.workflow)
        for new in added:
            await self._record(run_id, new.id, NodeState.PENDING)

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
        store the source of truth. Validation guarantees each source is an ancestor, so
        by the time this runs the output is there.
        """
        return {
            local_name: await self._store.load_output(run_id, source)
            for local_name, source in node.inputs.items()
        }

    # --- failure semantics ------------------------------------------------------------

    async def _apply_failure_mode(
        self,
        run_id: str,
        workflow: Workflow,
        failed: str,
        states: dict[str, NodeState],
        in_flight: dict[asyncio.Task[NodeState], str],
    ) -> bool:
        """React to one node's failure according to the run's failure mode (FR-5).

        Returns:
            Whether the loop should stop dispatching new nodes.
        """
        mode = self._policy.failure_mode
        if mode is FailureMode.RUN_TO_COMPLETION:
            return False

        if mode is FailureMode.SKIP_DOWNSTREAM:
            # Sorted so the record is written in the same order on every run, which is
            # what keeps two runs of the same failure byte-comparable.
            for node_id in sorted(descendants(workflow, failed)):
                # Absent means PENDING, the same rule `ready_set` and `_derive_run_state`
                # use. A node an expansion added is not in `states` until it is dispatched,
                # and reading a missing entry as "not pending" quietly excluded exactly the
                # nodes a grown graph most needs marking.
                if states.get(node_id, NodeState.PENDING) is not NodeState.PENDING:
                    continue
                states[node_id] = NodeState.SKIPPED
                await self._record(
                    run_id,
                    node_id,
                    NodeState.SKIPPED,
                    finished_at=self._clock.now(),
                    error=f"skipped: upstream node {failed!r} failed",
                )
            return False

        await self._cancel_siblings(run_id, in_flight, states, cause=failed)
        return True

    async def _cancel_siblings(
        self,
        run_id: str,
        in_flight: dict[asyncio.Task[NodeState], str],
        states: dict[str, NodeState],
        *,
        cause: str,
    ) -> None:
        """Stop every node still running, because one of their siblings failed.

        Safe to call with nothing in flight. Nothing has awaited between ``asyncio.wait``
        returning and this point, so every task here was genuinely still running when it
        was cancelled — but a task whose agent swallowed the cancellation and returned
        anyway keeps its real outcome, because reporting a completed node as cancelled
        would be exactly the kind of lie the store must not contain.
        """
        for task in in_flight:
            task.cancel()
        await asyncio.gather(*in_flight, return_exceptions=True)

        for task, node_id in in_flight.items():
            if not task.cancelled():
                states[node_id] = task.result()
                continue
            states[node_id] = NodeState.FAILED
            await self._record(
                run_id,
                node_id,
                NodeState.FAILED,
                finished_at=self._clock.now(),
                error=f"cancelled: fail_fast after node {cause!r} failed",
            )
        in_flight.clear()

    async def _cancel(self, in_flight: dict[asyncio.Task[NodeState], str]) -> None:
        """Cancel every in-flight node and wait for it to actually stop."""
        for task in in_flight:
            task.cancel()
        # gather() of nothing is a no-op, so this needs no empty check.
        await asyncio.gather(*in_flight, return_exceptions=True)
        in_flight.clear()

    # --- persistence ------------------------------------------------------------------

    async def _require_unused(self, run_id: str) -> None:
        """Refuse to start a run over the top of one that already exists.

        Before anything was durable this was harmless. Now it would overwrite a run that
        could have been resumed, which is a data-loss bug wearing the costume of a typo.

        Raises:
            ValidationError: If ``run_id`` is already in the store.
        """
        try:
            await self._store.load_run(run_id)
        except StoreError:
            return
        raise ValidationError(
            f"run {run_id!r} already exists; use resume({run_id!r}) to continue it, "
            "or choose a different run id"
        )

    async def _rehydrate_budget(self, run_id: str) -> None:
        """Charge the budget for everything this run spent before it was interrupted.

        Exact rather than approximate: the recorded responses carry their own token counts
        and are re-priced through the same :data:`~dagent.policy.limits.Pricer` a fresh run
        would use. A ceiling that reset itself on every crash would not be a ceiling.
        """
        for call in await self._store.load_model_calls(run_id):
            self._policy.budget.charge(
                tokens=call.response.total_tokens,
                cost_usd=self._policy.price(call.response),
            )

    async def _open_run(
        self, run_id: str, workflow: Workflow, states: Mapping[str, NodeState]
    ) -> None:
        """Write the definition, then the run and its initial node states.

        Definition first: a run record that exists must always have a definition behind
        it, or ``resume`` would find a run it cannot continue.
        """
        opened = self._clock.now()
        await self._store.save_workflow(run_id, workflow)
        await self._store.checkpoint(
            RunStateRecord(
                run_id=run_id,
                workflow_name=workflow.name,
                state=RunState.RUNNING,
                created_at=opened,
                updated_at=opened,
                nodes={
                    node_id: NodeStateRecord(run_id=run_id, node_id=node_id, state=state)
                    for node_id, state in states.items()
                },
            )
        )

    async def _checkpoint(self, run_id: str, state: RunState) -> RunStateRecord:
        """Persist the run's state, preserving everything written since it opened.

        Reloads before writing: node states have been written since the run was opened,
        and checkpointing a stale snapshot would throw them away.
        """
        run = await self._store.load_run(run_id)
        final = run.model_copy(update={"state": state, "updated_at": self._clock.now()})
        await self._store.checkpoint(final)
        return final

    async def _record(
        self,
        run_id: str,
        node_id: str,
        state: NodeState,
        *,
        attempt: int = 0,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        error: str | None = None,
    ) -> None:
        """Persist one node state transition."""
        await self._store.save_node_state(
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

    def _derive_run_state(self, workflow: Workflow, states: Mapping[str, NodeState]) -> RunState:
        """Turn the final node states, and the budget, into the run's outcome.

        The budget comes first: if it refused work, that is *why* the run ended the way it
        did, and ``FAILED`` would report the symptom instead of the cause. A ceiling merely
        crossed by the last call of an otherwise complete run is not a refusal and does not
        change a success into a failure — see :attr:`~dagent.policy.limits.Budget.refused`.

        A node blocked by a failed dependency is left ``PENDING`` unless the failure mode
        said otherwise, so it counts as "not SUCCESS" and the run is ``FAILED``.
        """
        if self._policy.budget.refused:
            return RunState.BUDGET_EXCEEDED
        # Over the graph's nodes, not over `states`: a node added by an expansion and then
        # never dispatched is absent from `states`, and a run is not a success because
        # nobody got round to one of its nodes.
        if all(
            states.get(node.id, NodeState.PENDING) is NodeState.SUCCESS for node in workflow.nodes
        ):
            return RunState.SUCCEEDED
        return RunState.FAILED
