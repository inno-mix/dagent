import pytest

from dagent.models.workflow import Node, Workflow


@pytest.fixture
def diamond() -> Workflow:
    """A -> B, A -> C, B+C -> D — the shape Phase 2's executor test will run."""
    return Workflow(
        name="diamond",
        nodes=(
            Node(id="a", agent="fake"),
            Node(id="b", agent="fake", depends_on=("a",), inputs={"x": "a"}),
            Node(id="c", agent="fake", depends_on=("a",), inputs={"x": "a"}),
            Node(id="d", agent="fake", depends_on=("b", "c"), inputs={"l": "b", "r": "c"}),
        ),
    )
