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
from dagent.policy.run import FailureMode, RunPolicy
from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import Clock, SystemClock
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
        return await self._drive(run_id, workflow, states, attempts={})

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
        return await self._drive(run_id, workflow, states, attempts)

    async def _drive(
        self,
        run_id: str,
        workflow: Workflow,
        states: dict[str, NodeState],
        attempts: Mapping[str, int],
    ) -> RunStateRecord:
        """The run loop, shared by a fresh run and a resumed one.

        There is exactly one of these on purpose: a resume that went down a second code
        path would be a second set of scheduling bugs, and the whole claim of DR-4 is that
        resume is *not* special — it is the same loop over reloaded state.
        """
        nodes = nodes_by_id(workflow)
        # Declaration order, so a batch of completions is folded in the same sequence
        # every time — see the comment where `declared` is used below.
        declared = {node_id: index for index, node_id in enumerate(nodes)}

        in_flight: dict[asyncio.Task[NodeState], str] = {}
        halted = False
        try:
            while True:
                if not halted:
                    for node_id in ready_set(workflow, states):
                        # Mark READY before dispatching: ready_set only offers PENDING
                        # nodes, so this is what stops the next pass re-dispatching one.
                        states[node_id] = NodeState.READY
                        await self._record(
                            run_id, node_id, NodeState.READY, attempt=attempts.get(node_id, 0)
                        )
                        task = asyncio.create_task(
                            self._execute(
                                run_id, nodes[node_id], first_attempt=attempts.get(node_id, 0)
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
                    halted |= await self._apply_failure_mode(
                        run_id, workflow, node_id, states, in_flight
                    )
        except asyncio.CancelledError:
            # The caller walked away: stop every node, then say so in the record rather
            # than leaving a run that looks like it is still going.
            await self._cancel(in_flight)
            await self._checkpoint(run_id, RunState.CANCELLED)
            raise
        except BaseException:
            await self._cancel(in_flight)
            raise

        return await self._checkpoint(run_id, self._derive_run_state(states))

    # --- one node -------------------------------------------------------------------

    async def _execute(self, run_id: str, node: Node, *, first_attempt: int = 0) -> NodeState:
        """Run one node under its own task, retrying as its policy allows.

        Args:
            run_id: The run this node belongs to.
            node: The node to execute.
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

        while True:
            await self._record(
                run_id, node.id, NodeState.RUNNING, attempt=attempt, started_at=started
            )
            try:
                output = await self._attempt(run_id, node, attempt=attempt, policy=policy)
            except Exception as exc:
                # A failing agent is an ordinary outcome, recorded and folded back into
                # the loop. CancelledError is not an Exception, so a cancelled node — and
                # therefore a timed-out one, once its timeout has converted it — still
                # propagates rather than being mistaken for a failure to retry.
                if attempt + 1 >= policy.max_attempts or not self._policy.retryable(exc):
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

                # Deliberately no FAILED write between attempts. FAILED is terminal, and a
                # terminal state a node then leaves is a lie — Phase 5's resume reads these
                # records to decide what still needs doing. The rising `attempt` on the
                # RUNNING record is the visible evidence that a retry happened.
                await self._clock.sleep(self._policy.backoff.delay(policy, attempt))
                attempt += 1
                continue

            # Output first, then SUCCESS: a node found SUCCESS must always have an output
            # to read, or Phase 5's resume would skip a node whose result it cannot
            # recover.
            await self._store.append_output(run_id, node.id, output)
            await self._record(
                run_id,
                node.id,
                NodeState.SUCCESS,
                attempt=attempt,
                started_at=started,
                finished_at=self._clock.now(),
            )
            return NodeState.SUCCESS

    async def _attempt(
        self, run_id: str, node: Node, *, attempt: int, policy: Policy
    ) -> NodeOutput:
        """Run the agent once, under this attempt's permits and timeout."""
        agent = self._registry.create(node.agent)
        inputs = await self._resolve_inputs(run_id, node)
        context = AgentContext(
            run_id=run_id,
            node_id=node.id,
            attempt=attempt,
            inputs=inputs,
            params=node.params,
            clock=self._clock,
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
            return await agent.run(context)

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
                if states.get(node_id) is not NodeState.PENDING:
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

    def _derive_run_state(self, states: Mapping[str, NodeState]) -> RunState:
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
        if all(state is NodeState.SUCCESS for state in states.values()):
            return RunState.SUCCEEDED
        return RunState.FAILED
