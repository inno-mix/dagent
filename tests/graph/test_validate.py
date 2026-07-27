import pytest

from dagent.errors import DagentError, ValidationError
from dagent.graph.topo import topological_order
from dagent.graph.validate import validate
from dagent.models.workflow import Node, Workflow


def test_a_valid_diamond_passes(diamond: Workflow) -> None:
    validate(diamond)


def test_a_single_node_workflow_passes() -> None:
    validate(Workflow(name="single", nodes=(Node(id="a", agent="fake"),)))


def test_an_empty_workflow_is_rejected() -> None:
    # Fail loud at submit time (AGENTS.md rule 6) rather than start a run that does nothing.
    with pytest.raises(ValidationError, match="declares no nodes"):
        validate(Workflow(name="empty", nodes=()))


def test_duplicate_node_ids_are_rejected() -> None:
    workflow = Workflow(
        name="duplicate",
        nodes=(Node(id="a", agent="fake"), Node(id="a", agent="fake")),
    )

    with pytest.raises(ValidationError, match="twice"):
        validate(workflow)


def test_a_dangling_dependency_is_rejected() -> None:
    workflow = Workflow(
        name="dangling",
        nodes=(Node(id="a", agent="fake", depends_on=("ghost",)),),
    )

    with pytest.raises(ValidationError, match="ghost"):
        validate(workflow)


def test_an_input_reading_from_a_node_that_does_not_exist_is_rejected() -> None:
    # Phase 1 acceptance: "a node referencing a non-existent upstream output is rejected".
    workflow = Workflow(
        name="unknown-source",
        nodes=(
            Node(id="a", agent="fake"),
            Node(id="b", agent="fake", depends_on=("a",), inputs={"x": "ghost"}),
        ),
    )

    with pytest.raises(ValidationError, match="ghost"):
        validate(workflow)


def test_an_input_reading_from_a_node_it_does_not_wait_for_is_rejected() -> None:
    # 'c' reads a's output but only waits for b, so a may still be PENDING when c runs.
    workflow = Workflow(
        name="race",
        nodes=(
            Node(id="a", agent="fake"),
            Node(id="b", agent="fake"),
            Node(id="c", agent="fake", depends_on=("b",), inputs={"x": "a"}),
        ),
    )

    with pytest.raises(ValidationError, match="race"):
        validate(workflow)


def test_an_input_reading_from_a_direct_dependency_is_accepted() -> None:
    workflow = Workflow(
        name="direct",
        nodes=(
            Node(id="a", agent="fake"),
            Node(id="b", agent="fake", depends_on=("a",), inputs={"x": "a"}),
        ),
    )

    validate(workflow)


def test_an_input_reading_from_a_transitive_ancestor_is_accepted() -> None:
    # c waits for b, b waits for a, so a is guaranteed finished before c reads it.
    workflow = Workflow(
        name="transitive",
        nodes=(
            Node(id="a", agent="fake"),
            Node(id="b", agent="fake", depends_on=("a",)),
            Node(id="c", agent="fake", depends_on=("b",), inputs={"x": "a"}),
        ),
    )

    validate(workflow)


def test_an_unregistered_agent_is_rejected_when_a_registry_is_supplied() -> None:
    workflow = Workflow(name="unknown-agent", nodes=(Node(id="a", agent="nope"),))

    with pytest.raises(ValidationError, match="not registered"):
        validate(workflow, known_agents={"fake", "researcher"})


def test_a_registered_agent_passes() -> None:
    workflow = Workflow(name="known-agent", nodes=(Node(id="a", agent="fake"),))

    validate(workflow, known_agents={"fake"})


def test_the_agent_check_is_skipped_when_no_registry_is_supplied() -> None:
    # dagent.graph must not depend on dagent.runtime, so the registry is injected and
    # validation stays usable with no registry in existence (Phase 1 has none).
    validate(Workflow(name="no-registry", nodes=(Node(id="a", agent="anything"),)))


def test_declaration_order_need_not_be_topological() -> None:
    # Nodes may be declared before the nodes they depend on; only the edges matter.
    workflow = Workflow(
        name="reversed",
        nodes=(
            Node(id="c", agent="fake", depends_on=("b",), inputs={"x": "b"}),
            Node(id="b", agent="fake", depends_on=("a",), inputs={"x": "a"}),
            Node(id="a", agent="fake"),
        ),
    )

    validate(workflow)
    assert topological_order(workflow) == ("a", "b", "c")


def test_every_rejection_is_catchable_as_a_dagent_error() -> None:
    with pytest.raises(DagentError):
        validate(Workflow(name="empty", nodes=()))


def test_validation_does_not_mutate_the_workflow(diamond: Workflow) -> None:
    before = diamond.model_dump()

    validate(diamond)

    assert diamond.model_dump() == before


def test_checks_run_in_a_fixed_order_so_the_message_is_deterministic() -> None:
    # This workflow is wrong three ways at once; the reported error must not vary.
    workflow = Workflow(
        name="multiply-broken",
        nodes=(
            Node(id="a", agent="unregistered", depends_on=("ghost",)),
            Node(id="a", agent="unregistered"),
        ),
    )

    for _ in range(3):
        with pytest.raises(ValidationError, match="twice"):
            validate(workflow, known_agents={"fake"})
