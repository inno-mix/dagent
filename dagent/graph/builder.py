"""A typed, fluent builder for constructing and validating a ``Workflow``.

The builder is the ergonomic front door; :func:`dagent.graph.validate.validate` is the
strict one. The builder adds each input's source node to that node's ``depends_on`` so a
caller does not have to say the same thing twice — but it relaxes no rule. The validator
still runs on what it produces, and a hand-written ``Workflow`` is held to exactly the
same standard.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Self

from pydantic import ValidationError as PydanticValidationError

from dagent.errors import ValidationError
from dagent.graph.validate import validate
from dagent.models.state import NodeOutput
from dagent.models.workflow import Node, Policy, Workflow

__all__ = ["WorkflowBuilder", "build_node"]


class WorkflowBuilder:
    """Accumulate node definitions, then :meth:`build` a frozen, validated ``Workflow``."""

    def __init__(self, name: str) -> None:
        """Start an empty workflow with the given name."""
        self._name = name
        self._nodes: list[Node] = []

    def add_node(
        self,
        node_id: str,
        agent: str,
        *,
        depends_on: Sequence[str] = (),
        inputs: Mapping[str, str] | None = None,
        params: Mapping[str, NodeOutput] | None = None,
        policy: Policy | None = None,
    ) -> Self:
        """Add a node, returning ``self`` so calls chain.

        Each input's source is appended to ``depends_on`` if it is not already there: a
        node that reads another node's output must also wait for it, and making the
        builder do that is what stops the ergonomic path from producing a race.

        Args:
            node_id: Unique id for the node within this workflow.
            agent: The registered agent name that will execute it.
            depends_on: Nodes to wait for whose output this node does not read.
            inputs: Local name → the id of the upstream node supplying it.
            params: Static configuration for the node, written by the author rather than
                produced upstream.
            policy: Optional per-node retry/timeout override.

        Returns:
            This builder.

        Raises:
            ValidationError: If the node itself is malformed. The graph-level rules are
                checked later, by :meth:`build`.
        """
        self._nodes.append(
            build_node(
                node_id,
                agent,
                depends_on=depends_on,
                inputs=inputs,
                params=params,
                policy=policy,
            )
        )
        return self

    def build(self, *, known_agents: Collection[str] | None = None) -> Workflow:
        """Freeze the accumulated nodes into a validated ``Workflow``.

        Args:
            known_agents: Registered agent names, forwarded to the validator. Omit to
                skip the agent check.

        Returns:
            A frozen, fully validated workflow.

        Raises:
            ValidationError: If the definition is malformed or the graph is invalid.
                Pydantic's own error is converted here so a caller only ever has to catch
                :class:`~dagent.errors.DagentError`.
        """
        try:
            workflow = Workflow(name=self._name, nodes=tuple(self._nodes))
        except PydanticValidationError as exc:
            raise ValidationError(f"invalid workflow {self._name!r}: {exc}") from exc

        validate(workflow, known_agents=known_agents)
        return workflow


def build_node(
    node_id: str,
    agent: str,
    *,
    depends_on: Sequence[str] = (),
    inputs: Mapping[str, str] | None = None,
    params: Mapping[str, NodeOutput] | None = None,
    policy: Policy | None = None,
) -> Node:
    """Build one node, adding each input's source to ``depends_on``.

    The same ergonomics as :meth:`WorkflowBuilder.add_node`, for callers that produce
    nodes without producing a whole workflow — a Phase 6 planner emitting nodes into a
    graph that is already running. Sharing the function is what stops the two paths from
    drifting: a planner that had to remember to declare its own edges would eventually
    forget, and the strict validator would reject its work at run time instead of at
    review time.

    Raises:
        ValidationError: If the node is malformed. Graph-level rules are someone else's
            job — :func:`dagent.graph.validate.validate` for a whole workflow, or
            :meth:`~dagent.runtime.expansion.RunGraph.apply` for an expansion.
    """
    resolved_inputs = dict(inputs or {})
    edges = list(depends_on)
    edges.extend(source for source in resolved_inputs.values() if source not in edges)

    try:
        return Node(
            id=node_id,
            agent=agent,
            depends_on=tuple(edges),
            inputs=resolved_inputs,
            params=dict(params or {}),
            policy=policy,
        )
    except PydanticValidationError as exc:
        raise ValidationError(f"invalid node {node_id!r}: {exc}") from exc
