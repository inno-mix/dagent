"""Phase 1 acceptance, stated as properties over randomly generated graphs.

Generation trick: name nodes ``n0..nk`` and only ever let a node depend on a
lower-numbered one. That makes acyclicity a property of the construction rather than
something the generator has to check, so the DAG strategy can never accidentally emit a
cyclic graph and quietly weaken the test.
"""

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dagent.errors import ValidationError
from dagent.graph.topo import node_dependencies, ready_set, topological_order
from dagent.graph.validate import validate
from dagent.models.state import NodeState
from dagent.models.workflow import Node, Workflow

MAX_NODES = 8


@st.composite
def dags(draw: st.DrawFn, *, min_size: int = 1, min_edges_per_node: int = 0) -> Workflow:
    """A random acyclic workflow of between ``min_size`` and ``MAX_NODES`` nodes."""
    size = draw(st.integers(min_value=min_size, max_value=MAX_NODES))
    ids = [f"n{index}" for index in range(size)]

    nodes: list[Node] = [Node(id=ids[0], agent="fake")]
    for index in range(1, size):
        dependencies = draw(
            st.lists(
                st.sampled_from(ids[:index]),
                unique=True,
                min_size=min(min_edges_per_node, index),
                max_size=index,
            )
        )
        nodes.append(Node(id=ids[index], agent="fake", depends_on=tuple(dependencies)))
    return Workflow(name="generated", nodes=tuple(nodes))


@given(dags())
def test_a_random_acyclic_graph_always_validates(workflow: Workflow) -> None:
    validate(workflow)


@given(dags())
def test_a_random_acyclic_graph_always_topologically_sorts(workflow: Workflow) -> None:
    order = topological_order(workflow)
    position = {node_id: index for index, node_id in enumerate(order)}

    assert set(order) == {node.id for node in workflow.nodes}
    assert len(order) == len(workflow.nodes)
    for node in workflow.nodes:
        for dependency in node_dependencies(node):
            assert position[dependency] < position[node.id]


@given(dags())
def test_the_ready_set_eventually_schedules_every_node(workflow: Workflow) -> None:
    """No static DAG can deadlock or starve: draining the ready set reaches every node."""
    states = {node.id: NodeState.PENDING for node in workflow.nodes}
    scheduled: list[str] = []

    while ready := ready_set(workflow, states):
        scheduled.extend(ready)
        for node_id in ready:
            states[node_id] = NodeState.SUCCESS

    assert set(scheduled) == {node.id for node in workflow.nodes}
    assert len(scheduled) == len(workflow.nodes), "a node was dispatched more than once"


@given(dags())
def test_the_ready_set_never_offers_a_node_before_its_dependencies(workflow: Workflow) -> None:
    states = {node.id: NodeState.PENDING for node in workflow.nodes}
    finished: set[str] = set()

    while ready := ready_set(workflow, states):
        by_id = {node.id: node for node in workflow.nodes}
        for node_id in ready:
            assert node_dependencies(by_id[node_id]) <= finished
            states[node_id] = NodeState.SUCCESS
        finished.update(ready)


@given(st.data(), dags(min_size=2, min_edges_per_node=1))
def test_a_back_edge_is_always_rejected_and_the_reported_cycle_is_real(
    data: st.DataObject, workflow: Workflow
) -> None:
    """Close any existing edge into a loop; validation must name a genuine cycle."""
    child = data.draw(st.sampled_from([node for node in workflow.nodes if node.depends_on]))
    parent_id = data.draw(st.sampled_from(child.depends_on))

    cyclic = Workflow(
        name=workflow.name,
        nodes=tuple(
            Node(id=node.id, agent=node.agent, depends_on=(*node.depends_on, child.id))
            if node.id == parent_id
            else node
            for node in workflow.nodes
        ),
    )

    with pytest.raises(ValidationError) as raised:
        validate(cyclic)

    cycle = raised.value.cycle
    assert cycle is not None, "a cycle rejection must carry the cycle path"
    assert cycle[0] == cycle[-1], f"{cycle} is not closed"
    assert len(cycle) >= 2

    edges = {node.id: node_dependencies(node) for node in cyclic.nodes}
    for source, target in itertools.pairwise(cycle):
        assert target in edges[source], f"{source} -> {target} is not an edge in the graph"


@given(st.data(), dags(min_size=2, min_edges_per_node=1))
def test_a_back_edge_is_also_rejected_by_the_topological_sort(
    data: st.DataObject, workflow: Workflow
) -> None:
    child = data.draw(st.sampled_from([node for node in workflow.nodes if node.depends_on]))
    parent_id = data.draw(st.sampled_from(child.depends_on))

    cyclic = Workflow(
        name=workflow.name,
        nodes=tuple(
            Node(id=node.id, agent=node.agent, depends_on=(*node.depends_on, child.id))
            if node.id == parent_id
            else node
            for node in workflow.nodes
        ),
    )

    with pytest.raises(ValidationError, match="cycle"):
        topological_order(cyclic)


@given(dags())
def test_validation_is_deterministic(workflow: Workflow) -> None:
    # Same input, same verdict, every time — a flaky validator makes a flaky test suite.
    def verdict() -> str | None:
        try:
            validate(workflow)
        except ValidationError as exc:  # pragma: no cover - generated graphs are valid
            return str(exc)
        return None

    assert verdict() == verdict()
