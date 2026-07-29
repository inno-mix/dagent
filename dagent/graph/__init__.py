"""Pure functions over a workflow: validation, topological order, ready set, builder.

Nothing in this package is async and nothing here touches I/O — ``tests/graph/
test_purity.py`` enforces that mechanically, because it is what keeps the graph layer
property-testable and the whole engine replayable.
"""

from dagent.graph.builder import WorkflowBuilder, build_node
from dagent.graph.expansion import expand_workflow
from dagent.graph.topo import (
    descendants,
    node_dependencies,
    nodes_by_id,
    ready_set,
    topological_order,
)
from dagent.graph.validate import find_cycle, validate

__all__ = [
    "WorkflowBuilder",
    "build_node",
    "descendants",
    "expand_workflow",
    "find_cycle",
    "node_dependencies",
    "nodes_by_id",
    "ready_set",
    "topological_order",
    "validate",
]
