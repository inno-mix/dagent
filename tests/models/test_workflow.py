import pytest
from pydantic import ValidationError as PydanticValidationError

from dagent.models.workflow import Node, Policy, Workflow


def _workflow() -> Workflow:
    return Workflow(name="w", nodes=(Node(id="a", agent="fake"),))


def test_workflow_nodes_are_a_tuple() -> None:
    # A list would let the runtime mutate a definition in place; FR-1 forbids that.
    assert isinstance(_workflow().nodes, tuple)


def test_workflow_is_frozen() -> None:
    workflow = _workflow()

    with pytest.raises(PydanticValidationError):
        workflow.name = "renamed"  # type: ignore[misc]


def test_node_is_frozen() -> None:
    node = Node(id="a", agent="fake")

    with pytest.raises(PydanticValidationError):
        node.agent = "other"  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    policy = Policy()

    with pytest.raises(PydanticValidationError):
        policy.max_attempts = 5  # type: ignore[misc]


@pytest.mark.parametrize("bad_id", ["", "1leading_digit", "has space", "has/slash", "a" * 129])
def test_node_id_rejects_identifiers_that_would_be_unsafe_as_keys(bad_id: str) -> None:
    with pytest.raises(PydanticValidationError):
        Node(id=bad_id, agent="fake")


@pytest.mark.parametrize("good_id", ["a", "_private", "research_1", "step.one", "step-two"])
def test_node_id_accepts_ordinary_identifiers(good_id: str) -> None:
    assert Node(id=good_id, agent="fake").id == good_id


def test_node_defaults_have_no_edges() -> None:
    node = Node(id="a", agent="fake")

    assert node.depends_on == ()
    assert dict(node.inputs) == {}
    assert node.policy is None


def test_node_requires_an_agent_name() -> None:
    with pytest.raises(PydanticValidationError):
        Node(id="a", agent="")


def test_models_reject_unknown_fields() -> None:
    # Fail loud at submit time (AGENTS.md rule 6): a typo'd key is a bug, not a no-op.
    with pytest.raises(PydanticValidationError):
        Node(id="a", agent="fake", dependson=("b",))  # type: ignore[call-arg]


def test_workflow_requires_a_name() -> None:
    with pytest.raises(PydanticValidationError):
        Workflow(name="", nodes=())


def test_policy_defaults_to_a_single_attempt_and_no_timeout() -> None:
    policy = Policy()

    assert policy.max_attempts == 1
    assert policy.timeout_s is None


def test_policy_rejects_zero_attempts() -> None:
    with pytest.raises(PydanticValidationError):
        Policy(max_attempts=0)


def test_policy_rejects_a_backoff_ceiling_below_its_floor() -> None:
    with pytest.raises(PydanticValidationError):
        Policy(backoff_initial_s=5.0, backoff_max_s=1.0)


def test_policy_rejects_a_non_positive_timeout() -> None:
    with pytest.raises(PydanticValidationError):
        Policy(timeout_s=0)


def test_a_node_can_carry_a_policy_override() -> None:
    node = Node(id="a", agent="fake", policy=Policy(max_attempts=3, timeout_s=2.5))

    assert node.policy is not None
    assert node.policy.max_attempts == 3
