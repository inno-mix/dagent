"""Merging new nodes into an existing workflow — the pure half of FR-7.

One function, and it is a pure one: given a workflow and some nodes, return the augmented
workflow or reject the whole thing. No run state, no scheduling, no I/O — which is what
lets ``tests/graph/test_purity.py`` cover it, and what makes the awkward case (a cycle
that only appears once two separate expansions are combined) cheap to test.

The bounds that need to know what a run is *doing* — how deep it has already expanded, how
big the graph is allowed to get — belong to :class:`~dagent.runtime.expansion.RunGraph`,
which calls this. Keeping them out of here is what keeps this function total.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from dagent.errors import ValidationError
from dagent.graph.topo import nodes_by_id
from dagent.graph.validate import validate
from dagent.models.workflow import Node, Workflow

__all__ = ["expand_workflow"]


def expand_workflow(
    workflow: Workflow,
    added: Sequence[Node],
    *,
    known_agents: Collection[str] | None = None,
) -> Workflow:
    """Append nodes to a workflow and revalidate the result.

    **Append only.** Existing nodes are carried over untouched, and that is the guardrail
    rather than an implementation detail: a node that is already ``RUNNING`` or ``SUCCESS``
    never re-checks its dependencies, so an expansion able to hand it a *new* one would
    strand it forever. A new node depending on an existing one is always safe — including
    on one that has already finished, which is exactly how a planner's generated children
    depend on the planner that generated them.

    Restating a node the workflow already has is a **no-op when the definition is
    identical**, and rejected when it differs. That is what lets a planner be re-executed
    after a crash (DR-4): it replays its own expansion, and applying it a second time
    changes nothing.

    Args:
        workflow: The existing definition. Never mutated — a new one is returned.
        added: Nodes to append, in order. Build them with
            :func:`~dagent.graph.builder.build_node`, so their ``depends_on`` edges match
            their inputs and the validator's ancestor rule is satisfied by construction.
        known_agents: Registered agent names, forwarded to the validator.

    Returns:
        The augmented, fully validated workflow — or the original object unchanged when
        ``added`` contributes nothing new.

    Raises:
        ValidationError: If a node redefines an existing one, or the augmented graph is
            invalid: a cycle, an unregistered agent, an input with no such upstream.
            Nothing is merged in that case, so the caller still holds a valid workflow.
    """
    existing = nodes_by_id(workflow)
    fresh: dict[str, Node] = {}

    for node in added:
        known = existing.get(node.id) or fresh.get(node.id)
        if known is None:
            fresh[node.id] = node
        elif known != node:
            raise ValidationError(
                f"expansion redefines node {node.id!r}; an expansion may add nodes but "
                "never change one that already exists"
            )

    if not fresh:
        return workflow

    augmented = workflow.model_copy(update={"nodes": (*workflow.nodes, *fresh.values())})
    validate(augmented, known_agents=known_agents)
    return augmented
