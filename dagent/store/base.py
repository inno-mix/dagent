"""The ``StateStore`` protocol: every durability semantic in dagent lives behind this.

The executor is written against this protocol and never against a concrete store, which
is what lets v1 run with zero external services and v2 gain durability by swapping the
implementation rather than rewriting the scheduler (DR-3).

Two ordering rules an implementation may rely on, and the executor guarantees:

* a node's output is written *before* that node is marked ``SUCCESS``, so a node found
  ``SUCCESS`` on reload always has an output to read;
* run-level state is written before the first node starts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dagent.models.state import NodeOutput, NodeStateRecord, RunStateRecord

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

    async def save_node_state(self, record: NodeStateRecord) -> None:
        """Write one node's state, replacing any previous state for that node."""
        ...

    async def append_output(self, run_id: str, node_id: str, output: NodeOutput) -> None:
        """Record the output produced by a node's latest attempt."""
        ...

    async def load_run(self, run_id: str) -> RunStateRecord:
        """Return a run's state, including every node's state."""
        ...

    async def load_output(self, run_id: str, node_id: str) -> NodeOutput:
        """Return the output a node produced.

        Reading outputs back through the store, rather than caching them in the executor,
        is what keeps the store the single source of truth — and is what makes Phase 5's
        resume a matter of reloading rather than reconstructing.
        """
        ...
