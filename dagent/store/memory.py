"""A dict-backed ``StateStore`` — the v1 implementation.

Deliberately the whole persistence story for v1: single-process, zero external services,
and fast enough that the test suite runs the real store rather than a mock. Postgres
arrives in Phase 5 behind the same protocol (DR-3).
"""

from __future__ import annotations

from collections.abc import Sequence

from dagent.errors import StoreError
from dagent.models.model_call import ModelCallRecord
from dagent.models.state import NodeOutput, NodeStateRecord, RunStateRecord
from dagent.models.workflow import Workflow

__all__ = ["InMemoryStateStore"]


class InMemoryStateStore:
    """Run state and outputs held in plain dicts.

    Not thread-safe, and does not need to be: v1 is a single event loop, and every
    method here completes without awaiting, so no other task can interleave inside one.
    """

    def __init__(self) -> None:
        """Start empty."""
        self._runs: dict[str, RunStateRecord] = {}
        self._workflows: dict[str, Workflow] = {}
        self._outputs: dict[tuple[str, str], NodeOutput] = {}
        self._model_calls: dict[str, dict[tuple[str, int, int], ModelCallRecord]] = {}

    async def checkpoint(self, run: RunStateRecord) -> None:
        """Write run-level state, replacing the stored record entirely."""
        self._runs[run.run_id] = run

    async def save_workflow(self, run_id: str, workflow: Workflow) -> None:
        """Write the definition this run is executing."""
        self._workflows[run_id] = workflow

    async def save_node_state(self, record: NodeStateRecord) -> None:
        """Write one node's state into its run."""
        run = self._require_run(record.run_id)
        nodes = {**run.nodes, record.node_id: record}
        self._runs[record.run_id] = run.model_copy(update={"nodes": nodes})

    async def append_output(self, run_id: str, node_id: str, output: NodeOutput) -> None:
        """Record the output produced by a node's latest attempt."""
        self._require_run(run_id)
        self._outputs[run_id, node_id] = output

    async def load_run(self, run_id: str) -> RunStateRecord:
        """Return a run's state."""
        return self._require_run(run_id)

    async def load_workflow(self, run_id: str) -> Workflow:
        """Return the definition a run is executing."""
        try:
            return self._workflows[run_id]
        except KeyError:
            raise StoreError(f"run {run_id!r} has no stored workflow") from None

    async def load_output(self, run_id: str, node_id: str) -> NodeOutput:
        """Return the output a node produced."""
        try:
            return self._outputs[run_id, node_id]
        except KeyError:
            raise StoreError(f"run {run_id!r} has no output for node {node_id!r}") from None

    async def append_model_call(self, record: ModelCallRecord) -> None:
        """Record one model call, keyed so a repeat of the same call replaces it."""
        self._require_run(record.run_id)
        calls = self._model_calls.setdefault(record.run_id, {})
        calls[record.node_id, record.attempt, record.sequence] = record

    async def load_model_calls(self, run_id: str) -> Sequence[ModelCallRecord]:
        """Return every model call made during a run, ordered by its key."""
        self._require_run(run_id)
        calls = self._model_calls.get(run_id, {})
        return tuple(calls[key] for key in sorted(calls))

    def _require_run(self, run_id: str) -> RunStateRecord:
        try:
            return self._runs[run_id]
        except KeyError:
            raise StoreError(f"unknown run {run_id!r}") from None
