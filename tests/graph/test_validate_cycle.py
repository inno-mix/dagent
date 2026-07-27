import pytest

from dagent.errors import ValidationError
from dagent.graph.builder import WorkflowBuilder
from dagent.graph.validate import find_cycle, validate
from dagent.models.workflow import Node, Workflow


def test_find_cycle_returns_none_for_an_acyclic_graph(diamond: Workflow) -> None:
    assert find_cycle(diamond) is None


def test_a_two_node_cycle_is_reported_with_its_path() -> None:
    workflow = Workflow(
        name="pair",
        nodes=(
            Node(id="a", agent="fake", depends_on=("b",)),
            Node(id="b", agent="fake", depends_on=("a",)),
        ),
    )

    with pytest.raises(ValidationError) as raised:
        validate(workflow)

    assert raised.value.cycle == ("a", "b", "a")
    assert "a -> b -> a" in str(raised.value)


def test_a_self_dependency_is_reported_as_a_one_node_cycle() -> None:
    workflow = Workflow(name="self", nodes=(Node(id="a", agent="fake", depends_on=("a",)),))

    with pytest.raises(ValidationError) as raised:
        validate(workflow)

    assert raised.value.cycle == ("a", "a")


def test_a_cycle_expressed_only_through_inputs_is_caught() -> None:
    # The builder turns each input into an edge, so a cycle can hide in inputs alone.
    builder = (
        WorkflowBuilder("via-inputs")
        .add_node("a", "fake", inputs={"x": "b"})
        .add_node("b", "fake", inputs={"x": "a"})
    )

    with pytest.raises(ValidationError) as raised:
        builder.build()

    assert raised.value.cycle is not None


def test_a_node_reached_from_an_earlier_root_is_not_searched_twice() -> None:
    # 'b' is declared before the 'a' it depends on, so the scan reaches 'a' first as a
    # dependency and must skip it when it comes up as a root.
    workflow = Workflow(
        name="reversed",
        nodes=(
            Node(id="b", agent="fake", depends_on=("a",)),
            Node(id="a", agent="fake"),
        ),
    )

    assert find_cycle(workflow) is None


def test_a_longer_cycle_names_every_member() -> None:
    workflow = Workflow(
        name="triangle",
        nodes=(
            Node(id="a", agent="fake", depends_on=("c",)),
            Node(id="b", agent="fake", depends_on=("a",)),
            Node(id="c", agent="fake", depends_on=("b",)),
        ),
    )

    with pytest.raises(ValidationError) as raised:
        validate(workflow)

    cycle = raised.value.cycle
    assert cycle is not None
    assert set(cycle) == {"a", "b", "c"}
    assert cycle[0] == cycle[-1]


def test_a_cycle_reachable_only_from_a_later_root_is_still_found() -> None:
    # The search must not stop after the first acyclic component.
    workflow = Workflow(
        name="late",
        nodes=(
            Node(id="clean", agent="fake"),
            Node(id="a", agent="fake", depends_on=("b",)),
            Node(id="b", agent="fake", depends_on=("a",)),
        ),
    )

    with pytest.raises(ValidationError) as raised:
        validate(workflow)

    assert raised.value.cycle is not None
    assert "clean" not in raised.value.cycle


def test_a_cycle_that_dangles_off_an_acyclic_prefix_is_found() -> None:
    workflow = Workflow(
        name="tail",
        nodes=(
            Node(id="root", agent="fake"),
            Node(id="a", agent="fake", depends_on=("root", "b")),
            Node(id="b", agent="fake", depends_on=("a",)),
        ),
    )

    with pytest.raises(ValidationError) as raised:
        validate(workflow)

    assert set(raised.value.cycle or ()) == {"a", "b"}


def test_the_reported_cycle_is_deterministic() -> None:
    workflow = Workflow(
        name="triangle",
        nodes=(
            Node(id="a", agent="fake", depends_on=("c",)),
            Node(id="b", agent="fake", depends_on=("a",)),
            Node(id="c", agent="fake", depends_on=("b",)),
        ),
    )

    assert len({find_cycle(workflow) for _ in range(5)}) == 1


def test_a_rejection_that_is_not_a_cycle_leaves_the_path_unset() -> None:
    # So a caller can tell a cycle rejection from any other kind.
    with pytest.raises(ValidationError) as raised:
        validate(Workflow(name="empty", nodes=()))

    assert raised.value.cycle is None
