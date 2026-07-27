import pytest

from dagent.errors import ValidationError
from dagent.graph.topo import node_dependencies, nodes_by_id, ready_set, topological_order
from dagent.models.state import NodeState
from dagent.models.workflow import Node, Workflow


def _all_pending(workflow: Workflow) -> dict[str, NodeState]:
    return {node.id: NodeState.PENDING for node in workflow.nodes}


# --- node_dependencies -------------------------------------------------------------


def test_dependencies_are_the_declared_edges() -> None:
    node = Node(id="d", agent="fake", depends_on=("a", "b", "c"))

    assert node_dependencies(node) == frozenset({"a", "b", "c"})


def test_an_input_does_not_implicitly_create_an_edge() -> None:
    # If it did, the ancestor rule could never be violated and FR-2's "unsatisfied
    # input" rejection would be unreachable. WorkflowBuilder declares the edge instead.
    node = Node(id="b", agent="fake", inputs={"x": "a"})

    assert node_dependencies(node) == frozenset()


def test_a_node_named_in_both_depends_on_and_inputs_is_counted_once() -> None:
    node = Node(id="b", agent="fake", depends_on=("a",), inputs={"x": "a"})

    assert node_dependencies(node) == frozenset({"a"})


def test_a_node_with_no_edges_depends_on_nothing() -> None:
    assert node_dependencies(Node(id="a", agent="fake")) == frozenset()


# --- nodes_by_id -------------------------------------------------------------------


def test_nodes_by_id_preserves_declaration_order(diamond: Workflow) -> None:
    assert list(nodes_by_id(diamond)) == ["a", "b", "c", "d"]


# --- ready_set ---------------------------------------------------------------------


def test_only_the_sources_are_ready_at_the_start(diamond: Workflow) -> None:
    assert ready_set(diamond, _all_pending(diamond)) == ("a",)


def test_the_graph_fans_out_once_the_source_succeeds(diamond: Workflow) -> None:
    states = _all_pending(diamond)
    states["a"] = NodeState.SUCCESS

    assert ready_set(diamond, states) == ("b", "c")


def test_fan_in_waits_for_every_branch(diamond: Workflow) -> None:
    states = _all_pending(diamond)
    states["a"] = NodeState.SUCCESS
    states["b"] = NodeState.RUNNING
    states["c"] = NodeState.RUNNING

    assert ready_set(diamond, states) == ()

    states["b"] = NodeState.SUCCESS

    assert ready_set(diamond, states) == (), "d must not start while c is still running"

    states["c"] = NodeState.SUCCESS

    assert ready_set(diamond, states) == ("d",)


@pytest.mark.parametrize(
    "blocking_state",
    [NodeState.PENDING, NodeState.READY, NodeState.RUNNING, NodeState.FAILED, NodeState.SKIPPED],
)
def test_a_dependency_that_has_not_succeeded_blocks_readiness(
    diamond: Workflow, blocking_state: NodeState
) -> None:
    states = _all_pending(diamond)
    states["a"] = blocking_state

    assert "b" not in ready_set(diamond, states)


@pytest.mark.parametrize(
    "dispatched_state",
    [NodeState.READY, NodeState.RUNNING, NodeState.SUCCESS, NodeState.FAILED, NodeState.SKIPPED],
)
def test_a_node_that_has_left_pending_is_never_offered_again(
    diamond: Workflow, dispatched_state: NodeState
) -> None:
    # Otherwise the executor would dispatch the same node on every loop iteration.
    states = _all_pending(diamond)
    states["a"] = dispatched_state

    assert "a" not in ready_set(diamond, states)


def test_a_node_missing_from_states_counts_as_pending(diamond: Workflow) -> None:
    # Phase 6 inserts expanded nodes without backfilling state first.
    assert ready_set(diamond, {}) == ("a",)


def test_ready_set_returns_declaration_order_not_sorted_order() -> None:
    # Determinism (AGENTS.md rule 4) means reproducing the author's order, not alphabetising.
    workflow = Workflow(
        name="independent",
        nodes=(
            Node(id="zebra", agent="fake"),
            Node(id="mango", agent="fake"),
            Node(id="apple", agent="fake"),
        ),
    )

    assert ready_set(workflow, _all_pending(workflow)) == ("zebra", "mango", "apple")


def test_ready_set_is_stable_across_calls(diamond: Workflow) -> None:
    states = _all_pending(diamond)
    states["a"] = NodeState.SUCCESS

    assert ready_set(diamond, states) == ready_set(diamond, states)


def test_a_terminal_graph_offers_nothing(diamond: Workflow) -> None:
    states = {node.id: NodeState.SUCCESS for node in diamond.nodes}

    assert ready_set(diamond, states) == ()


# --- topological_order -------------------------------------------------------------


def test_topological_order_lists_every_node(diamond: Workflow) -> None:
    assert set(topological_order(diamond)) == {"a", "b", "c", "d"}


def test_topological_order_places_each_node_after_its_dependencies(diamond: Workflow) -> None:
    order = topological_order(diamond)
    position = {node_id: index for index, node_id in enumerate(order)}

    for node in diamond.nodes:
        for dependency in node_dependencies(node):
            assert position[dependency] < position[node.id]


def test_topological_order_breaks_ties_by_declaration_order(diamond: Workflow) -> None:
    assert topological_order(diamond) == ("a", "b", "c", "d")


def test_topological_order_is_stable_across_calls(diamond: Workflow) -> None:
    assert topological_order(diamond) == topological_order(diamond)


def test_topological_order_rejects_a_cycle() -> None:
    workflow = Workflow(
        name="cyclic",
        nodes=(
            Node(id="a", agent="fake", depends_on=("b",)),
            Node(id="b", agent="fake", depends_on=("a",)),
        ),
    )

    with pytest.raises(ValidationError, match="cycle"):
        topological_order(workflow)


def test_topological_order_rejects_a_dangling_dependency() -> None:
    workflow = Workflow(
        name="dangling",
        nodes=(Node(id="a", agent="fake", depends_on=("ghost",)),),
    )

    with pytest.raises(ValidationError, match="ghost"):
        topological_order(workflow)
