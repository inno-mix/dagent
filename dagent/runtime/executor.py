"""The scheduler: the ready-set run loop that drives a workflow to completion.

The whole engine is one loop — compute the ready set, dispatch it, wait for the first
completion, fold the result in, repeat. Fan-out is dispatching several ready nodes at
once; fan-in is the recomputation afterwards noticing that a node's last dependency just
landed. Nothing here knows what an agent does, and nothing here imports a model SDK.

Policy wraps that loop rather than complicating it. A node's attempts, its timeout, the
permits it holds and the budget it spends are all decided in ``dagent.policy`` and merely
*applied* by the runner, which is why adding retries did not change the loop by a line.

Nor did distributing it. This class is the coordinator in both v1 and v2; the only thing
that varies is the :class:`~dagent.runtime.dispatch.Dispatcher` it was handed, and with it
whether a ready node becomes a task on this event loop or a message another process picks
up (DR-12). Everything below — validation, the ready set, the failure modes, the run-state
derivation, resume — is the same code either way, and that unchanged core is the whole
point of the exercise.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from dagent.errors import StoreError, ValidationError
from dagent.graph.topo import descendants, nodes_by_id, ready_set
from dagent.graph.validate import validate
from dagent.models.state import (
    NodeState,
    NodeStateRecord,
    RunState,
    RunStateRecord,
)
from dagent.models.workflow import Workflow
from dagent.observability import logging as obs_logging
from dagent.observability import metrics as obs_metrics
from dagent.observability import tracing
from dagent.policy.run import FailureMode, RunPolicy
from dagent.runtime.clock import Clock, SystemClock
from dagent.runtime.dispatch import Dispatcher, LocalDispatcher
from dagent.runtime.expansion import RunGraph
from dagent.runtime.model import ModelClient, NullModelClient
from dagent.runtime.node import NodeOutcome, NodeRunner, record_state
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
    """Runs a validated workflow to completion, on one event loop or across many.

    Everything the executor depends on is injected: the agents come from a registry, the
    durability from a ``StateStore`` protocol, time from a ``Clock``, every limit from a
    ``RunPolicy``, and — since Phase 8 — the transport from a ``Dispatcher``. That is what
    keeps this class testable with no network, no database, and no waiting.
    """

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        store: StateStore,
        clock: Clock | None = None,
        model: ModelClient | None = None,
        policy: RunPolicy | None = None,
        dispatcher: Dispatcher | None = None,
    ) -> None:
        """Wire the executor to its agents, storage, source of time, provider, and limits.

        One ``model`` client is shared by the whole run (AGENTS.md §4: never a client per
        call). Each node gets thin recording and metering wrappers around it, not a
        connection of its own. Omit it and agents that try to call a model fail with a
        clear message.

        Omitting ``policy`` gives a run with no caps, no ceiling, one attempt per node and
        no timeout — the behaviour of an executor with no policy layer at all, so a limit
        is always something a caller asked for.

        Omitting ``dispatcher`` runs every node as a task on this event loop, which is what
        the executor did before there was a choice. Passing a
        :class:`~dagent.runtime.dispatch.QueueDispatcher` makes this process a coordinator
        and leaves the running of nodes to workers; ``registry`` and ``model`` are then
        only consulted for validation, because nothing here will execute an agent.
        """
        self._registry = registry
        self._store = store
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._model: ModelClient = model if model is not None else NullModelClient()
        self._policy = policy if policy is not None else RunPolicy()
        self._dispatcher: Dispatcher = (
            dispatcher
            if dispatcher is not None
            else LocalDispatcher(
                NodeRunner(
                    registry=registry,
                    store=store,
                    clock=self._clock,
                    model=self._model,
                    policy=self._policy,
                ),
                store=store,
            )
        )
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
        started = self._clock.now()
        log = obs_logging.get_logger()
        try:
            with (
                obs_logging.bind_run(run_id, graph.workflow.name),
                tracing.run_span(run_id, graph.workflow.name) as span,
            ):
                log.info("run.started", nodes=len(graph.workflow.nodes))
                state = await self._loop(run_id, graph, states, attempts, log)
                span.set_attribute(tracing.RUN_STATE, state.value)
        except asyncio.CancelledError:
            # The caller walked away: stop every node, then say so in the record rather
            # than leaving a run that looks like it is still going.
            await self._dispatcher.abandon()
            await self._checkpoint(run_id, RunState.CANCELLED)
            raise
        except BaseException:
            await self._dispatcher.abandon()
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
        log: Any,
    ) -> RunState:
        """Drive the ready set until no further progress is possible.

        Split out from :meth:`_drive` so the run span and the log binding wrap it as one
        block rather than being threaded through the loop body. Cancellation handling
        stays outside, where it can still checkpoint after the context managers unwind.
        """
        dispatcher = self._dispatcher
        halted = False
        # Whatever a previous coordinator of this run was told and never acted on. Always
        # empty for a fresh run and for an in-process dispatcher; on resume across a queue
        # it is the set that must be folded in before anything else, because a resumed run
        # has nothing outstanding and would otherwise be declared finished on the spot —
        # taking an unmerged expansion, and the work it described, with it.
        recovered = await dispatcher.start(run_id, graph)
        while True:
            # Derived once per graph version rather than per pass: an expansion
            # replaces the workflow object, so the index invalidates itself by identity.
            # Rebuilding these every pass is what made the loop quadratic in node count.
            index = graph.index
            workflow, nodes, declared = index.workflow, index.nodes, index.declared

            if recovered:
                for node_id in await self._fold(run_id, graph, recovered, declared, states):
                    log.warning("node.failed", node_id=node_id)
                    halted |= await self._apply_failure_mode(run_id, graph, node_id, states)
                recovered = ()
                # Round again rather than on: folding may have grown the graph, and the
                # index above describes the version from before it did.
                continue

            ready = () if halted else ready_set(workflow, states, dependencies=index.dependencies)
            self._metrics.ready_set_size.record(len(ready), {obs_metrics.WORKFLOW: workflow.name})
            for node_id in ready:
                # Mark READY before dispatching: ready_set only offers PENDING
                # nodes, so this is what stops the next pass re-dispatching one.
                states[node_id] = NodeState.READY
                attempt = attempts.get(node_id, 0)
                await self._record(run_id, node_id, NodeState.READY, attempt=attempt)
                await dispatcher.dispatch(nodes[node_id], attempt=attempt)

            # Nothing outstanding and nothing ready: the graph can make no more progress.
            # With a failed node that leaves its dependents PENDING — see _derive_run_state.
            if not dispatcher.outstanding:
                break

            for node_id in await self._fold(
                run_id, graph, await dispatcher.settle(), declared, states
            ):
                log.warning("node.failed", node_id=node_id)
                halted |= await self._apply_failure_mode(run_id, graph, node_id, states)

        return self._derive_run_state(graph.workflow, states)

    async def _fold(
        self,
        run_id: str,
        graph: RunGraph,
        outcomes: Sequence[NodeOutcome],
        declared: Mapping[str, int],
        states: dict[str, NodeState],
    ) -> list[str]:
        """Absorb a batch of completions into the run's state, and name the failures.

        Several nodes can land in one batch, and a transport is free to report them in
        whatever order they arrived. Folding them in arrival order would make the record
        depend on timing — two identical runs could then attribute the same skipped node to
        different upstream failures. Declaration order restores rule 4, and it is the same
        tiebreak ``ready_set`` uses.
        """
        failures: list[str] = []
        for outcome in sorted(outcomes, key=lambda finished: declared[finished.node_id]):
            states[outcome.node_id] = await self._absorb(run_id, graph, outcome)
            if states[outcome.node_id] is NodeState.FAILED:
                failures.append(outcome.node_id)
        # Only now: a result is released once the coordinator has acted on it, so a crash in
        # between leaves it to be delivered again rather than losing what it was carrying.
        await self._dispatcher.acknowledge(outcomes)
        return failures

    async def _absorb(self, run_id: str, graph: RunGraph, outcome: NodeOutcome) -> NodeState:
        """Fold one completed node into the run, growing the graph if it asked to.

        An outcome only carries an expansion when the node ran somewhere that does not own
        the graph. In a single process the sink merged it before the node was ever marked
        ``SUCCESS``; here the merge happens after, which is why the result is not released
        until it is done.

        A rejected expansion turns its node ``FAILED``, which is the same verdict a
        single-process run reaches — one hop later, and for the same reason: the request
        was checked against a graph nobody else was editing. A worker could not have made
        that call, because the copy it holds may already be out of date.
        """
        if not outcome.expansion:
            return outcome.state

        try:
            added = graph.apply(outcome.node_id, outcome.expansion)
        except ValidationError as exc:
            await self._record(
                run_id,
                outcome.node_id,
                NodeState.FAILED,
                attempt=outcome.attempt,
                finished_at=self._clock.now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            return NodeState.FAILED

        if added:
            await self._store.save_workflow(run_id, graph.workflow)
            for node in added:
                await self._record(run_id, node.id, NodeState.PENDING)
        return outcome.state

    # --- failure semantics ------------------------------------------------------------

    async def _apply_failure_mode(
        self,
        run_id: str,
        graph: RunGraph,
        failed: str,
        states: dict[str, NodeState],
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
            for node_id in sorted(descendants(graph.workflow, failed)):
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

        await self._cancel_siblings(run_id, graph, states, cause=failed)
        return True

    async def _cancel_siblings(
        self,
        run_id: str,
        graph: RunGraph,
        states: dict[str, NodeState],
        *,
        cause: str,
    ) -> None:
        """Stop every node still running, because one of their siblings failed.

        Safe to call with nothing outstanding. What "stop" means is the dispatcher's to
        decide — cancelled on this event loop, merely awaited across processes — but the
        rule here is the same either way: a node that reached a verdict of its own keeps
        it, because reporting a completed node as cancelled would be exactly the kind of
        lie the store must not contain.
        """
        outcomes = await self._dispatcher.halt()
        for outcome in sorted(outcomes, key=lambda stopped: graph.index.declared[stopped.node_id]):
            if not outcome.cancelled:
                states[outcome.node_id] = await self._absorb(run_id, graph, outcome)
                continue
            states[outcome.node_id] = NodeState.FAILED
            await self._record(
                run_id,
                outcome.node_id,
                NodeState.FAILED,
                attempt=outcome.attempt,
                finished_at=self._clock.now(),
                error=f"cancelled: fail_fast after node {cause!r} failed",
            )

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

        Distributing the run does not distribute this. Workers spend against their own
        in-process budgets and the coordinator holds the run's, which means a token ceiling
        is enforced per worker rather than per run — the one policy that a shared-nothing
        transport genuinely weakens, and it is stated here rather than left to be
        discovered.
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
        it, or ``resume`` would find a run it cannot continue — and, since Phase 8, a
        worker handed one of its nodes would find nothing to look the node up in.
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
        await record_state(
            self._store,
            run_id,
            node_id,
            state,
            attempt=attempt,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
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
