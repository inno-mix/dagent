"""Submit-time validation of a workflow definition.

Validation is pure, total, and fast (FR-2): a definition either fully validates or is
rejected, before anything executes and before any side effect is possible. Checks run in
a fixed order and each reports the first problem it finds, so the error a caller sees for
a given workflow is always the same one — a flaky error message is a flaky test.
"""

from __future__ import annotations

from collections.abc import Collection

from dagent.errors import ValidationError
from dagent.graph.topo import node_dependencies, nodes_by_id, topological_order
from dagent.models.workflow import Workflow

__all__ = ["find_cycle", "validate"]

_WHITE, _GREY, _BLACK = 0, 1, 2


def validate(workflow: Workflow, *, known_agents: Collection[str] | None = None) -> None:
    """Raise :class:`~dagent.errors.ValidationError` if this workflow could never run.

    Checks, in order: the workflow declares nodes, node ids are unique, every agent is
    registered, every referenced node exists, the graph is acyclic, and every input reads
    from an ancestor.

    Args:
        workflow: The definition to check. Never mutated.
        known_agents: The registered agent names. Omit it, or pass ``None``, to skip the
            agent check. The registry is *injected* rather than imported because
            ``dagent.graph`` must not depend on ``dagent.runtime`` — that keeps this
            module pure and testable with no registry in existence at all.

    Raises:
        ValidationError: On the first problem found. A cycle rejection also carries the
            cycle path on the exception's ``cycle`` attribute.
    """
    _check_declares_nodes(workflow)
    _check_ids_are_unique(workflow)
    _check_agents_are_known(workflow, known_agents)
    _check_references_exist(workflow)
    _check_acyclic(workflow)
    _check_inputs_are_satisfied(workflow)


def find_cycle(workflow: Workflow) -> tuple[str, ...] | None:
    """Return one dependency cycle as a path, or ``None`` if the graph is acyclic.

    Depth-first search with three-colour marking: grey means "on the current path", so
    an edge into a grey node closes a cycle. Roots are visited in declaration order and
    each node's edges in sorted order, so the cycle reported for a given graph is always
    the same one. Iterative rather than recursive so a deep chain cannot exhaust the
    Python stack.

    Returns:
        A path where each element depends on the next and the first and last elements are
        the same node — ``("a", "b", "a")`` reads "a depends on b depends on a". A node
        that depends on itself is reported as ``("a", "a")``.
    """
    nodes = nodes_by_id(workflow)
    edges = {
        node_id: tuple(sorted(dep for dep in node_dependencies(node) if dep in nodes))
        for node_id, node in nodes.items()
    }
    colour = dict.fromkeys(nodes, _WHITE)

    for root in nodes:
        if colour[root] != _WHITE:
            continue
        colour[root] = _GREY
        path = [root]
        # Each frame is (node, index of the next edge to walk).
        stack = [(root, 0)]
        while stack:
            node_id, edge_index = stack[-1]
            if edge_index == len(edges[node_id]):
                colour[node_id] = _BLACK
                stack.pop()
                path.pop()
                continue

            stack[-1] = (node_id, edge_index + 1)
            dependency = edges[node_id][edge_index]
            if colour[dependency] == _GREY:
                return (*path[path.index(dependency) :], dependency)
            if colour[dependency] == _WHITE:
                colour[dependency] = _GREY
                path.append(dependency)
                stack.append((dependency, 0))
    return None


def _check_declares_nodes(workflow: Workflow) -> None:
    if not workflow.nodes:
        raise ValidationError(f"workflow {workflow.name!r} declares no nodes")


def _check_ids_are_unique(workflow: Workflow) -> None:
    seen: set[str] = set()
    for node in workflow.nodes:
        if node.id in seen:
            raise ValidationError(f"workflow {workflow.name!r} declares node {node.id!r} twice")
        seen.add(node.id)


def _check_agents_are_known(workflow: Workflow, known_agents: Collection[str] | None) -> None:
    if known_agents is None:
        return
    known = frozenset(known_agents)
    for node in workflow.nodes:
        if node.agent not in known:
            raise ValidationError(
                f"node {node.id!r} names agent {node.agent!r}, which is not registered"
            )


def _check_references_exist(workflow: Workflow) -> None:
    ids = frozenset(node.id for node in workflow.nodes)
    for node in workflow.nodes:
        # Inputs first: the builder copies each input's source into depends_on, so a typo
        # shows up in both places and the input-specific message is the more useful one.
        for name, source in node.inputs.items():
            if source not in ids:
                raise ValidationError(
                    f"node {node.id!r} input {name!r} reads from {source!r}, "
                    "which is not in this workflow"
                )
        for dependency in node.depends_on:
            if dependency not in ids:
                raise ValidationError(
                    f"node {node.id!r} depends on {dependency!r}, which is not in this workflow"
                )


def _check_acyclic(workflow: Workflow) -> None:
    cycle = find_cycle(workflow)
    if cycle is not None:
        raise ValidationError(
            f"workflow {workflow.name!r} contains a cycle: {' -> '.join(cycle)}",
            cycle=cycle,
        )


def _check_inputs_are_satisfied(workflow: Workflow) -> None:
    """Reject an input that reads from a node its reader does not wait for.

    That is the "unsatisfied input" of FR-2: the output may not exist yet when the reader
    runs, so it is a race rather than a dependency. Ancestor sets are accumulated in
    topological order, which needs one pass because every dependency is finished before
    the node that uses it is reached.
    """
    nodes = nodes_by_id(workflow)
    ancestors: dict[str, frozenset[str]] = {}
    for node_id in topological_order(workflow):
        reachable: set[str] = set()
        for dependency in node_dependencies(nodes[node_id]):
            reachable.add(dependency)
            reachable |= ancestors[dependency]
        ancestors[node_id] = frozenset(reachable)

    for node in workflow.nodes:
        for name, source in node.inputs.items():
            if source not in ancestors[node.id]:
                raise ValidationError(
                    f"node {node.id!r} input {name!r} reads from {source!r}, which is not "
                    f"upstream of {node.id!r}; add it to depends_on or the read is a race"
                )
