"""The scheduler: the ready-set run loop that drives a workflow to completion.

The whole engine is one loop — compute the ready set, dispatch it, wait for the first
completion, fold the result in, repeat. Fan-out is dispatching several ready nodes at
once; fan-in is the recomputation afterwards noticing that a node's last dependency just
landed. Nothing here knows what an agent does, and nothing here imports a model SDK.

Phase 2 scope: a static graph, no retries, no timeouts, no concurrency caps. Those are
the policy layer's job (Phase 4) and slot in around ``_execute`` without touching the
loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime

from dagent.graph.topo import nodes_by_id, ready_set
from dagent.graph.validate import validate
from dagent.models.state import (
    NodeOutput,
    NodeState,
    NodeStateRecord,
    RunState,
    RunStateRecord,
)
from dagent.models.workflow import Node, Workflow
from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import Clock, SystemClock
from dagent.runtime.registry import AgentRegistry
from dagent.store.base import StateStore

__all__ = ["Executor"]

_FIRST_ATTEMPT = 0
"""Phase 2 runs every node exactly once. Phase 4's retries are what make this vary."""


class Executor:
    """Runs a validated workflow to completion on a single event loop.

    Everything the executor depends on is injected: the agents come from a registry, the
    durability from a ``StateStore`` protocol, and time from a ``Clock``. That is what
    keeps this class testable with no network, no database, and no waiting.
    """

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        store: StateStore,
        clock: Clock | None = None,
    ) -> None:
        """Wire the executor to its agents, its storage, and its source of time."""
        self._registry = registry
        self._store = store
        self._clock: Clock = clock if clock is not None else SystemClock()

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
            ValidationError: If the workflow is invalid or names an unregistered agent.
                Nothing executes and nothing is written in that case.
        """
        # Submit-time validation, with the registry injected — this is the caller that
        # actually has one, which is why graph/ takes it as a parameter.
        validate(workflow, known_agents=self._registry.names())

        nodes = nodes_by_id(workflow)
        states: dict[str, NodeState] = dict.fromkeys(nodes, NodeState.PENDING)
        await self._open_run(run_id, workflow, states)

        in_flight: dict[asyncio.Task[NodeState], str] = {}
        try:
            while True:
                for node_id in ready_set(workflow, states):
                    # Mark READY before dispatching: ready_set only offers PENDING nodes,
                    # so this is what stops the next pass re-dispatching the same node.
                    states[node_id] = NodeState.READY
                    await self._record(run_id, node_id, NodeState.READY)
                    task = asyncio.create_task(
                        self._execute(run_id, nodes[node_id]),
                        name=f"dagent:{run_id}:{node_id}",
                    )
                    in_flight[task] = node_id

                # Nothing running and nothing ready: the graph can make no more progress.
                # With a failed node that leaves its dependents PENDING — see _close_run.
                if not in_flight:
                    break

                done, _ = await asyncio.wait(set(in_flight), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    states[in_flight.pop(task)] = task.result()
        except BaseException:
            # Includes CancelledError: if the caller walks away, no node keeps running.
            await self._cancel(in_flight)
            raise

        return await self._close_run(run_id, states)

    async def _execute(self, run_id: str, node: Node) -> NodeState:
        """Run one node under its own task and persist the outcome."""
        started = self._clock.now()
        await self._record(run_id, node.id, NodeState.RUNNING, started_at=started)

        try:
            agent = self._registry.create(node.agent)
            inputs = await self._resolve_inputs(run_id, node)
            output = await agent.run(
                AgentContext(
                    run_id=run_id,
                    node_id=node.id,
                    attempt=_FIRST_ATTEMPT,
                    inputs=inputs,
                    clock=self._clock,
                )
            )
        except Exception as exc:
            # A failing agent is an ordinary outcome, recorded and folded back into the
            # loop. CancelledError is not an Exception, so cancellation still propagates.
            await self._record(
                run_id,
                node.id,
                NodeState.FAILED,
                started_at=started,
                finished_at=self._clock.now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            return NodeState.FAILED

        # Output first, then SUCCESS: a node found SUCCESS must always have an output to
        # read, or Phase 5's resume would skip a node whose result it cannot recover.
        await self._store.append_output(run_id, node.id, output)
        await self._record(
            run_id,
            node.id,
            NodeState.SUCCESS,
            started_at=started,
            finished_at=self._clock.now(),
        )
        return NodeState.SUCCESS

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

    async def _open_run(
        self, run_id: str, workflow: Workflow, states: Mapping[str, NodeState]
    ) -> None:
        """Write the run and its initial node states before anything executes."""
        opened = self._clock.now()
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

    async def _close_run(self, run_id: str, states: Mapping[str, NodeState]) -> RunStateRecord:
        """Derive and persist the run's terminal state.

        Reloads before writing: node states have been written since the run was opened,
        and checkpointing a stale snapshot would throw them away.

        A node blocked by a failed dependency is left ``PENDING`` here. Marking it
        ``SKIPPED`` is one of the three failure semantics Phase 4 makes configurable, and
        picking one now would be inventing a policy this phase has no mandate for.
        """
        run = await self._store.load_run(run_id)
        final = run.model_copy(
            update={
                "state": _derive_run_state(states),
                "updated_at": self._clock.now(),
            }
        )
        await self._store.checkpoint(final)
        return final

    async def _record(
        self,
        run_id: str,
        node_id: str,
        state: NodeState,
        *,
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
                attempt=_FIRST_ATTEMPT,
                started_at=started_at,
                finished_at=finished_at,
                error=error,
            )
        )

    async def _cancel(self, in_flight: dict[asyncio.Task[NodeState], str]) -> None:
        """Cancel every in-flight node and wait for it to actually stop."""
        for task in in_flight:
            task.cancel()
        # gather() of nothing is a no-op, so this needs no empty check.
        await asyncio.gather(*in_flight, return_exceptions=True)
        in_flight.clear()


def _derive_run_state(states: Mapping[str, NodeState]) -> RunState:
    """A run succeeded only if every one of its nodes did.

    Phase 4 adds the other two terminal states: ``BUDGET_EXCEEDED`` when admission is
    refused, and ``CANCELLED`` when the run is stopped from outside.
    """
    if all(state == NodeState.SUCCESS for state in states.values()):
        return RunState.SUCCEEDED
    return RunState.FAILED
