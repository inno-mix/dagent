"""Run and node state: the two lifecycle enums and the records the store persists.

Data only — no logic. ``NodeState`` describes one node, ``RunState`` describes a whole
run. AGENTS.md §5 calls conflating those two the single most common bug in this codebase,
so they are deliberately separate types whose members only coincide where the meaning
genuinely does.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

__all__ = [
    "TERMINAL_NODE_STATES",
    "TERMINAL_RUN_STATES",
    "NodeOutput",
    "NodeState",
    "NodeStateRecord",
    "RunState",
    "RunStateRecord",
]

NodeOutput: TypeAlias = JsonValue
"""Whatever an agent returns.

Typed as JSON rather than ``Any`` because FR-4 requires a *serializable* output and
Phase 5 has to write these to Postgres. Making that a type error today is cheaper than
discovering it at the storage boundary later; an agent holding a richer object calls
``.model_dump()`` on the way out.
"""


class NodeState(StrEnum):
    """Where a single node is in its lifecycle within one run."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunState(StrEnum):
    """Where a whole run is in its lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"


TERMINAL_NODE_STATES: frozenset[NodeState] = frozenset(
    {NodeState.SUCCESS, NodeState.FAILED, NodeState.SKIPPED}
)
"""Node states no transition leaves."""

TERMINAL_RUN_STATES: frozenset[RunState] = frozenset(
    {
        RunState.SUCCEEDED,
        RunState.FAILED,
        RunState.BUDGET_EXCEEDED,
        RunState.CANCELLED,
    }
)
"""Run states no transition leaves."""


class NodeStateRecord(BaseModel):
    """One node's durable state within a run.

    ``attempt`` is carried from the very first phase so Phase 5's idempotency key
    ``(run_id, node_id, attempt)`` is derivable without a schema change — DR-4 puts
    idempotency before durability, and that starts with the record shape.

    Timestamps default to ``None`` rather than to the wall clock: AGENTS.md rule 4 puts
    every source of non-determinism behind an injected seam, so the executor fills these
    from the injected ``Clock`` and a replayed run reproduces them exactly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    node_id: str
    state: NodeState = NodeState.PENDING
    attempt: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class RunStateRecord(BaseModel):
    """A whole run's durable state, including a snapshot of each node's state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    workflow_name: str
    state: RunState = RunState.PENDING
    created_at: datetime | None = None
    updated_at: datetime | None = None
    nodes: Mapping[str, NodeStateRecord] = Field(default_factory=dict)
