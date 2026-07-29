"""Graph expansion and validation logic for dynamic DAG augmentation (Phase 6).

Pure functions that:
1. Validate an expansion before it happens (validate_expansion)
2. Augment a workflow with new nodes and revalidate (expand_workflow)

Both operations are total and deterministic: they either fully succeed or reject with
a specific error message. No side effects, no I/O, no randomness.
"""

from __future__ import annotations

from collections.abc import Collection

from dagent.errors import ValidationError
from dagent.graph.validate import validate
from dagent.models.expansion import NodeDefinition
from dagent.models.workflow import Node, Workflow

__all__ = ["expand_workflow", "validate_expansion"]


def expand_workflow(
    workflow: Workflow,
    new_nodes: list[NodeDefinition],
    *,
    known_agents: Collection[str] | None = None,
) -> Workflow:
    """Augment a workflow with new nodes and validate the result.

    Takes an existing Workflow and a list of NodeDefinitions, converts the
    NodeDefinitions to Node objects (inferring depends_on from inputs), and returns
    a new Workflow with both existing and new nodes. Calls validate() to ensure the
    augmented graph is acyclic, has no unknown agents, and all inputs are satisfied.

    Args:
        workflow: The existing workflow to augment. Never mutated.
        new_nodes: List of new nodes to add. Order is preserved.
        known_agents: Registered agent names. Pass None to skip agent validation.

    Returns:
        A new Workflow with existing nodes followed by new nodes, fully validated.

    Raises:
        ValidationError: If names collide, agents are unknown, or the augmented graph
            is invalid (cycle, dangling reference, unsatisfied input, etc).
    """
    # Check for name collisions
    existing_ids = frozenset(node.id for node in workflow.nodes)
    for node_def in new_nodes:
        if node_def.name in existing_ids:
            raise ValidationError(
                f"cannot expand workflow: new node {node_def.name!r} already exists"
            )

    # Check for duplicate names within new_nodes
    seen: set[str] = set()
    for node_def in new_nodes:
        if node_def.name in seen:
            raise ValidationError(
                f"cannot expand workflow: new node {node_def.name!r} declared twice"
            )
        seen.add(node_def.name)

    # Convert NodeDefinitions to Nodes, inferring depends_on from inputs
    converted_nodes: list[Node] = []
    for node_def in new_nodes:
        # depends_on = all nodes referenced in inputs
        depends_on = tuple(sorted(set(node_def.inputs.values())))
        node = Node(
            id=node_def.name,
            agent=node_def.agent,
            depends_on=depends_on,
            inputs=node_def.inputs,
            params=node_def.params,
            policy=node_def.policy,
        )
        converted_nodes.append(node)

    # Build augmented workflow
    augmented_nodes = (*workflow.nodes, *converted_nodes)
    augmented = Workflow(name=workflow.name, nodes=augmented_nodes)

    # Validate the augmented graph
    validate(augmented, known_agents=known_agents)

    return augmented


def validate_expansion(
    workflow: Workflow,
    new_nodes: list[NodeDefinition],
    running_nodes: set[str],
    expansion_count: int,
    max_depth: int,
    *,
    known_agents: Collection[str] | None = None,
) -> None:
    """Validate an expansion before it happens.

    Checks (in order):
    1. Expansion depth: expansion_count < max_depth
    2. Strand prevention: no new node depends on an already-SUCCESS node
    3. Agent validity: all agents are registered
    4. Reference validity: all input references exist in workflow or new_nodes

    Args:
        workflow: The current workflow.
        new_nodes: Nodes to be added.
        running_nodes: Nodes currently in execution (not yet SUCCESS).
        expansion_count: Number of expansions already done.
        max_depth: Maximum allowed expansion count.
        known_agents: Registered agent names. Pass None to skip agent validation.

    Raises:
        ValidationError: On the first check that fails.
    """
    # Check 1: Expansion depth
    if expansion_count >= max_depth:
        raise ValidationError(f"expansion depth limit exceeded: {expansion_count} >= {max_depth}")

    # Build sets for validation
    existing_ids = frozenset(node.id for node in workflow.nodes)
    new_ids = frozenset(node_def.name for node_def in new_nodes)
    # During execution, SUCCESS nodes = those not currently running. Any node not in
    # running_nodes is assumed to be SUCCESS (already executed and won't re-run).
    success_nodes = existing_ids - running_nodes
    available_nodes = existing_ids | new_ids

    # Check 2: Strand prevention - new nodes may not depend on SUCCESS nodes
    for node_def in new_nodes:
        for input_source in node_def.inputs.values():
            if input_source in success_nodes:
                raise ValidationError(
                    f"cannot expand workflow: new node {node_def.name!r} depends on "
                    f"{input_source!r}, which is already SUCCESS; this would create a "
                    "strand (a node depending on completed work cannot be replayed)"
                )

    # Check 3: Agent validity
    if known_agents is not None:
        known = frozenset(known_agents)
        for node_def in new_nodes:
            if node_def.agent not in known:
                raise ValidationError(
                    f"new node {node_def.name!r} names agent {node_def.agent!r}, "
                    "which is not registered"
                )

    # Check 4: Reference validity
    for node_def in new_nodes:
        for name, source in node_def.inputs.items():
            if source not in available_nodes:
                raise ValidationError(
                    f"new node {node_def.name!r} input {name!r} reads from {source!r}, "
                    "which is not in this workflow"
                )
