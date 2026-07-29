"""The ``StateStore`` protocol: every durability semantic in dagent lives behind this.

The executor is written against this protocol and never against a concrete store, which
is what lets v1 run with zero external services and v2 gain durability by swapping the
implementation rather than rewriting the scheduler (DR-3).

Three ordering rules an implementation may rely on, and the executor guarantees:

* the workflow definition is written before the run record, so a run that exists can
  always be resumed;
* run-level state is written before the first node starts;
* a node's output is written *before* that node is marked ``SUCCESS``, so a node found
  ``SUCCESS`` on reload always has an output to read.

Together those are what make :meth:`~dagent.runtime.executor.Executor.resume` a *reload*
rather than a reconstruction: everything needed to continue a run is in here, and the
executor keeps no state a crash could take with it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from dagent.models.model_call import ModelCallRecord
from dagent.models.state import NodeOutput, NodeStateRecord, RunStateRecord
from dagent.models.workflow import Workflow

__all__ = ["StateStore"]


@runtime_checkable
class StateStore(Protocol):
    """Durable storage for run state and node outputs.

    Every method raises :class:`~dagent.errors.StoreError` on a missing record rather
    than returning ``None``: a run or output the executor asks for and does not get is a
    consistency bug, not an ordinary outcome.
    """

    async def checkpoint(self, run: RunStateRecord) -> None:
        """Write run-level state, creating the run if this is its first write.

        Replaces the stored record wholesale, node map included. A caller updating a live
        run loads it first and writes back what it read, rather than assembling a record
        from memory that may have gone stale.
        """
        ...

    async def save_workflow(self, run_id: str, workflow: Workflow) -> None:
        """Write the definition this run is executing.

        Stored once, at submit time, and separately from the run record rather than inside
        it: ``checkpoint`` replaces the whole record on every run-level write, and
        re-serializing an entire graph each time would be a lot of bytes to say "this run
        is still going".

        Persisting the definition at all is what lets ``resume(run_id)`` take no other
        argument. Reconstructing it from the original file instead would work today and
        break in Phase 6, where a planner adds nodes at run time that no file contains.
        """
        ...

    async def save_node_state(self, record: NodeStateRecord) -> None:
        """Write one node's state, replacing any previous state for that node."""
        ...

    async def append_output(self, run_id: str, node_id: str, output: NodeOutput) -> None:
        """Record the output produced by a node's latest attempt."""
        ...

    async def load_run(self, run_id: str) -> RunStateRecord:
        """Return a run's state, including every node's state."""
        ...

    async def load_workflow(self, run_id: str) -> Workflow:
        """Return the definition a run is executing."""
        ...

    async def load_output(self, run_id: str, node_id: str) -> NodeOutput:
        """Return the output a node produced.

        Reading outputs back through the store, rather than caching them in the executor,
        is what keeps the store the single source of truth — and is what makes Phase 5's
        resume a matter of reloading rather than reconstructing.
        """
        ...

    async def append_model_call(self, record: ModelCallRecord) -> None:
        """Record one model call, keyed by ``(run_id, node_id, attempt, sequence)``.

        A node may call a model several times, and every one of them has to survive for a
        replay to reproduce the run (FR-8) — which is what the ``sequence`` component of
        the key is for. Recording the *same* key twice replaces rather than duplicates: a
        node re-executed at the same attempt after a crash replays its own sequence
        numbers, and two stored answers to one key is a replay that cannot choose.
        """
        ...

    async def load_model_calls(self, run_id: str) -> Sequence[ModelCallRecord]:
        """Return every model call made during a run, ordered by its key.

        Ordered by ``(node_id, attempt, sequence)`` rather than by wall-clock arrival.
        Under concurrency there is no single "order they were made" worth reproducing —
        two nodes interleave differently on every run — whereas ordering by the key is
        deterministic, identical across implementations, and is what a replay indexes on.
        """
        ...
