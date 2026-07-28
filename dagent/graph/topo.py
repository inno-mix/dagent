"""Dependency edges, topological order, and the ready set.

Pure functions over a ``Workflow``: no async, no I/O, no randomness. Every result is a
deterministic function of the workflow and the states handed in, which is what lets a
replayed run schedule its nodes in exactly the order the original did (AGENTS.md rule 4).
Where a tiebreak is needed, declaration order is the tiebreak.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from dagent.errors import ValidationError
from dagent.models.state import NodeState
from dagent.models.workflow import Node, Workflow

__all__ = [
    "descendants",
    "node_dependencies",
    "nodes_by_id",
    "ready_set",
    "topological_order",
]


def node_dependencies(node: Node) -> frozenset[str]:
    """Return every node id this node waits on: its ``depends_on`` edges.

    Reading an input does *not* implicitly create an edge here. If it did, an input could
    never fail the ancestor rule, FR-2's "unsatisfied input" rejection could never fire,
    and a requirement would be silently unimplemented. Instead the edge must be declared:
    ``WorkflowBuilder`` adds it for you, and ``dagent.graph.validate`` rejects any input
    whose source is not an ancestor. One definition of the edge set, used by the ready
    set, the topological order, and validation alike.
    """
    return frozenset(node.depends_on)


def nodes_by_id(workflow: Workflow) -> dict[str, Node]:
    """Index a workflow's nodes by id, preserving declaration order."""
    return {node.id: node for node in workflow.nodes}


def topological_order(workflow: Workflow) -> tuple[str, ...]:
    """Return every node id in dependency order.

    Kahn's algorithm over a queue seeded in declaration order, so the same workflow
    always produces the same sequence.

    Args:
        workflow: The definition to order. Never mutated.

    Returns:
        Node ids, each appearing after every node it depends on.

    Raises:
        ValidationError: If a dependency names a node that does not exist, or if the
            graph contains a cycle. Call ``dagent.graph.validate`` first for a message
            that names the offending cycle.
    """
    nodes = nodes_by_id(workflow)
    dependencies = {node_id: node_dependencies(node) for node_id, node in nodes.items()}

    unknown = {dep for deps in dependencies.values() for dep in deps if dep not in nodes}
    if unknown:
        raise ValidationError(f"unknown dependency: {', '.join(sorted(unknown))}")

    outstanding = {node_id: len(deps) for node_id, deps in dependencies.items()}
    dependents: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    # Iterating in declaration order keeps each dependents list in declaration order too,
    # which is what makes the queue — and therefore the result — reproducible.
    for node_id, deps in dependencies.items():
        for dep in deps:
            dependents[dep].append(node_id)

    queue = deque(node_id for node_id, count in outstanding.items() if count == 0)
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for dependent in dependents[node_id]:
            outstanding[dependent] -= 1
            if outstanding[dependent] == 0:
                queue.append(dependent)

    if len(order) != len(nodes):
        stuck = sorted(set(nodes) - set(order))
        raise ValidationError(
            f"workflow {workflow.name!r} contains a cycle among: {', '.join(stuck)}"
        )
    return tuple(order)


def descendants(workflow: Workflow, node_id: str) -> frozenset[str]:
    """Return every node that depends on ``node_id``, directly or transitively.

    This is the blast radius of a failure: once ``node_id`` fails, no node in this set can
    ever satisfy :func:`ready_set`, so the ``skip_downstream`` failure mode uses it to say
    so out loud rather than leaving those nodes ``PENDING`` forever.

    Iterative rather than recursive, for the same reason cycle detection is: a deep chain
    must not depend on the interpreter's stack limit. Tolerates a ``node_id`` that is not
    in the workflow by returning an empty set — a caller asking about a node that has just
    been removed wants an empty blast radius, not an exception.
    """
    dependents: dict[str, list[str]] = {node.id: [] for node in workflow.nodes}
    for node in workflow.nodes:
        for dependency in node_dependencies(node):
            if dependency in dependents:
                dependents[dependency].append(node.id)

    reached: set[str] = set()
    queue = deque(dependents.get(node_id, ()))
    while queue:
        current = queue.popleft()
        if current in reached:
            continue
        reached.add(current)
        queue.extend(dependents[current])
    return frozenset(reached)


def ready_set(workflow: Workflow, states: Mapping[str, NodeState]) -> tuple[str, ...]:
    """Return the ids of every node that can start now, in declaration order.

    A node is ready when it is still ``PENDING`` and every node it depends on is
    ``SUCCESS``. A node missing from ``states`` counts as ``PENDING``, so a graph that
    just grew (Phase 6) needs no state backfill before it can be scheduled.

    Ordered rather than an unordered set on purpose: dispatch order then reproduces on
    replay. Cost is O(nodes + edges), so recomputing after every completion is cheap.
    """
    ready: list[str] = []
    for node in workflow.nodes:
        if states.get(node.id, NodeState.PENDING) != NodeState.PENDING:
            continue
        if all(
            states.get(dependency, NodeState.PENDING) == NodeState.SUCCESS
            for dependency in node_dependencies(node)
        ):
            ready.append(node.id)
    return tuple(ready)
