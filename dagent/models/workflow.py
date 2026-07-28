"""Frozen workflow definition schemas. Data only — the rules live in ``dagent.graph``.

A definition is immutable once constructed (FR-1): the runtime never mutates it, and
Phase 6's dynamic expansion produces a *new* ``Workflow`` rather than editing one in
place. Keeping the schema free of validation logic is what lets ``dagent.graph`` stay a
set of pure functions that are trivially property-testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from dagent.models.state import NodeOutput

__all__ = ["Node", "NodeId", "Policy", "Workflow"]

NodeId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$", max_length=128),
]
"""A node identifier: identifier-shaped, so it is safe in logs, spans, and store keys."""


class Policy(BaseModel):
    """Per-node retry and timeout limits.

    Phase 1 only gives these a type; Phase 4 is what enforces them. Run-level concerns
    (budget ceilings, failure semantics) are deliberately not here — they belong to the
    run, not to a node.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=1, ge=1)
    backoff_initial_s: float = Field(default=0.1, gt=0)
    backoff_max_s: float = Field(default=30.0, gt=0)
    timeout_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_backoff_bounds(self) -> Policy:
        """Reject a ceiling below its floor, which would otherwise silently clamp."""
        if self.backoff_initial_s > self.backoff_max_s:
            raise ValueError(
                f"backoff_initial_s ({self.backoff_initial_s}) exceeds "
                f"backoff_max_s ({self.backoff_max_s})"
            )
        return self


class Node(BaseModel):
    """One unit of work: a registered agent plus the edges that feed and gate it.

    ``depends_on`` is the edge set, and the only thing the scheduler waits on.

    ``inputs`` maps a local name the agent will read to the id of the upstream node whose
    output supplies it. Naming a node here does *not* implicitly add an edge: an input may
    only read from an *ancestor*, and ``dagent.graph.validate`` rejects one that does not,
    because reading the output of a node you do not wait for is a race rather than a
    dependency. ``WorkflowBuilder`` adds the edge for you, so the ergonomic path cannot
    produce that race and the strict path still catches a hand-written one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: NodeId
    agent: str = Field(min_length=1)
    depends_on: tuple[NodeId, ...] = ()
    inputs: Mapping[str, NodeId] = Field(default_factory=dict)
    params: Mapping[str, NodeOutput] = Field(default_factory=dict)
    """Static configuration for this node, handed to the agent alongside its inputs.

    Where ``inputs`` carries data produced by other nodes, ``params`` carries data the
    author wrote down: the topic a researcher is asked about, a threshold, a style. It is
    part of the frozen definition, so it is identical on every replay — and it is how a
    Phase 6 planner will hand each generated node its own subtopic.
    """
    policy: Policy | None = None


class Workflow(BaseModel):
    """An immutable set of nodes.

    Declaration order is significant: it is the tiebreak that makes topological order and
    ready-set dispatch deterministic, and therefore replayable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    nodes: tuple[Node, ...]
