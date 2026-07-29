"""Node expansion schema for dynamic DAG expansion (Phase 6).

The planner agent returns these models to request new nodes be added to the workflow
at runtime. Both models are immutable (frozen) to maintain consistency with the
broader design principle that definitions never mutate.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from dagent.models.workflow import Policy

__all__ = ["ExpansionResult", "NodeDefinition"]


class NodeDefinition(BaseModel):
    """One new node to add to the workflow during dynamic expansion.

    This represents a node the planner has decided should exist, ready to be added to
    the graph via Phase 6's expansion mechanism. It is identical in structure to
    ``dagent.models.workflow.Node`` except it lacks ``id`` and ``depends_on`` (those
    are derived by the expansion validator) and uses a more lenient ``name`` field
    instead of the strict ``NodeId`` type (the planner may generate names that do not
    yet exist, and validation happens downstream).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    """Unique node name within the workflow. Validated by the graph builder."""

    agent: str = Field(min_length=1)
    """Registered agent name. Must match an entry in the agent registry."""

    inputs: dict[str, str] = Field(default_factory=dict)
    """Upstream node outputs: {param_name: node_id}.

    Maps local parameter names to the node IDs whose outputs should feed this node.
    The graph validator will add the corresponding edge and reject cycles.
    """

    params: dict[str, JsonValue] = Field(default_factory=dict)
    """Static parameters passed to the agent.

    These are frozen as part of the definition, so they are identical on every
    replay. The planner uses this to pass each generated node its own subtopic,
    threshold, or other configuration.
    """

    policy: Policy | None = None
    """Node-level retry and timeout policy override.

    If None, the node inherits the workflow's default policy. If set, this policy
    applies to this node alone.
    """


class ExpansionResult(BaseModel):
    """The output of a planner agent: new nodes to add to the workflow.

    The planner agent's run method returns one of these, signaling which nodes it
    has decided should be added during the current expansion phase. The executor then
    passes these to the graph builder for validation and incorporation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: list[NodeDefinition] = Field(default_factory=list)
    """List of new nodes to add to the workflow.

    Order is significant: nodes that appear first are added first, making the
    expansion deterministic and replayable. An empty list signals no expansion.
    """

    max_depth: int = Field(default=100, ge=1, le=1000)
    """Maximum expansion depth for this and subsequent phases.

    Limits the number of dynamic expansion passes to prevent runaway growth. Must
    be between 1 and 1000 (inclusive). Defaults to 100. This is a safeguard against
    infinite or excessive expansion loops.
    """
