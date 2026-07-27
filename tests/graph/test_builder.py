import pytest
from pydantic import ValidationError as PydanticValidationError

from dagent.errors import ValidationError
from dagent.graph.builder import WorkflowBuilder
from dagent.models.workflow import Policy, Workflow


def test_build_returns_a_frozen_workflow() -> None:
    workflow = WorkflowBuilder("single").add_node("a", "fake").build()

    assert isinstance(workflow, Workflow)
    assert workflow.name == "single"
    assert isinstance(workflow.nodes, tuple)


def test_add_node_chains() -> None:
    workflow = (
        WorkflowBuilder("chain")
        .add_node("a", "fake")
        .add_node("b", "fake", inputs={"x": "a"})
        .add_node("c", "fake", inputs={"x": "b"})
        .build()
    )

    assert [node.id for node in workflow.nodes] == ["a", "b", "c"]


def test_an_input_implies_a_dependency_edge() -> None:
    # This is what stops the ergonomic path from producing a read-before-write race.
    workflow = (
        WorkflowBuilder("implied")
        .add_node("a", "fake")
        .add_node("b", "fake", inputs={"x": "a"})
        .build()
    )

    assert workflow.nodes[1].depends_on == ("a",)


def test_an_explicit_dependency_is_not_duplicated() -> None:
    workflow = (
        WorkflowBuilder("dedupe")
        .add_node("a", "fake")
        .add_node("b", "fake", depends_on=["a"], inputs={"x": "a"})
        .build()
    )

    assert workflow.nodes[1].depends_on == ("a",)


def test_ordering_only_edges_survive_alongside_inputs() -> None:
    workflow = (
        WorkflowBuilder("mixed")
        .add_node("a", "fake")
        .add_node("b", "fake")
        .add_node("c", "fake", depends_on=["a"], inputs={"x": "b"})
        .build()
    )

    assert set(workflow.nodes[2].depends_on) == {"a", "b"}


def test_a_node_policy_is_carried_through() -> None:
    workflow = (
        WorkflowBuilder("policy")
        .add_node("a", "fake", policy=Policy(max_attempts=3, timeout_s=1.5))
        .build()
    )

    assert workflow.nodes[0].policy == Policy(max_attempts=3, timeout_s=1.5)


def test_build_validates_the_graph() -> None:
    builder = (
        WorkflowBuilder("cyclic")
        .add_node("a", "fake", depends_on=["b"])
        .add_node("b", "fake", depends_on=["a"])
    )

    with pytest.raises(ValidationError) as raised:
        builder.build()

    assert raised.value.cycle is not None


def test_build_rejects_an_empty_workflow() -> None:
    with pytest.raises(ValidationError, match="declares no nodes"):
        WorkflowBuilder("empty").build()


def test_build_forwards_the_registry_to_the_validator() -> None:
    builder = WorkflowBuilder("agents").add_node("a", "unregistered")

    with pytest.raises(ValidationError, match="not registered"):
        builder.build(known_agents={"fake"})


def test_build_accepts_a_registered_agent() -> None:
    WorkflowBuilder("agents").add_node("a", "fake").build(known_agents={"fake"})


def test_a_malformed_node_raises_a_dagent_error_not_a_pydantic_one() -> None:
    # Callers should only ever have to catch DagentError.
    with pytest.raises(ValidationError) as raised:
        WorkflowBuilder("bad").add_node("1leading-digit", "fake")

    assert not isinstance(raised.value, PydanticValidationError)


def test_a_malformed_workflow_name_raises_a_dagent_error() -> None:
    with pytest.raises(ValidationError) as raised:
        WorkflowBuilder("").add_node("a", "fake").build()

    assert not isinstance(raised.value, PydanticValidationError)


def test_the_builder_holds_a_hand_written_workflow_to_the_same_standard() -> None:
    # The builder is ergonomic, not lenient: it produces exactly what validate() accepts.
    built = (
        WorkflowBuilder("diamond")
        .add_node("a", "fake")
        .add_node("b", "fake", inputs={"x": "a"})
        .add_node("c", "fake", inputs={"x": "a"})
        .add_node("d", "fake", inputs={"l": "b", "r": "c"})
        .build()
    )

    assert [node.depends_on for node in built.nodes] == [(), ("a",), ("a",), ("b", "c")]
