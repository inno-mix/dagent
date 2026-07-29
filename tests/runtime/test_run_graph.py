"""`RunGraph`: the run-level expansion bounds, and the atomicity the design rests on.

Named for the class rather than the concept because `tests/graph/test_expansion.py`
already owns that basename, and pytest identifies test modules by basename alone.
"""

import ast
import inspect
import textwrap

import pytest

from dagent.errors import ValidationError
from dagent.graph.builder import build_node
from dagent.models.workflow import Workflow
from dagent.runtime.expansion import Expansion, RunGraph

KNOWN = {"fake"}


@pytest.fixture
def diamond() -> Workflow:
    """A -> B, A -> C, B+C -> D."""
    return Workflow(
        name="diamond",
        nodes=(
            build_node("a", "fake"),
            build_node("b", "fake", inputs={"x": "a"}),
            build_node("c", "fake", inputs={"x": "a"}),
            build_node("d", "fake", inputs={"l": "b", "r": "c"}),
        ),
    )


def base() -> Workflow:
    return Workflow(name="base", nodes=(build_node("root", "fake"),))


def asking_for(*nodes: object) -> Expansion:
    expansion = Expansion()
    expansion.add(*nodes)  # type: ignore[arg-type]
    return expansion


def graph_for(workflow: Workflow | None = None, **kwargs: object) -> RunGraph:
    return RunGraph(workflow or base(), known_agents=KNOWN, **kwargs)  # type: ignore[arg-type]


# --- the atomicity claim ---------------------------------------------------------------


def test_apply_contains_no_await() -> None:
    # The load-bearing property. `RunGraph.apply` runs inside a node's task, and two
    # planners can finish in the same batch. With no suspension point the whole
    # validate-and-merge is atomic on one event loop, so the second is checked against a
    # graph that already includes the first — no lock, and no window in which the graph is
    # half-expanded. A stray `await` here would reintroduce exactly that window, silently.
    tree = ast.parse(textwrap.dedent(inspect.getsource(RunGraph.apply)))
    suspensions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Await | ast.AsyncFor | ast.AsyncWith)
    ]

    assert not suspensions, "RunGraph.apply must stay synchronous — see the module docstring"


def test_a_second_expansion_sees_the_first(diamond: Workflow) -> None:
    graph = graph_for(diamond, max_depth=2)

    graph.apply("a", asking_for(build_node("e", "fake", inputs={"x": "a"})))
    graph.apply("b", asking_for(build_node("f", "fake", inputs={"y": "e"})))

    assert [node.id for node in graph.workflow.nodes] == ["a", "b", "c", "d", "e", "f"]


# --- ordinary use ----------------------------------------------------------------------


def test_the_graph_starts_as_the_workflow_it_was_given() -> None:
    workflow = base()

    assert graph_for(workflow).workflow is workflow


def test_apply_returns_the_nodes_it_added() -> None:
    graph = graph_for()

    added = graph.apply("root", asking_for(build_node("child", "fake", inputs={"x": "root"})))

    assert [node.id for node in added] == ["child"]


def test_an_empty_request_changes_nothing() -> None:
    graph = graph_for()
    before = graph.workflow

    assert graph.apply("root", Expansion()) == ()
    assert graph.workflow is before


def test_restating_an_existing_node_adds_nothing() -> None:
    graph = graph_for()
    before = graph.workflow

    assert graph.apply("root", asking_for(*before.nodes)) == ()
    assert graph.workflow is before


def test_an_expansion_object_is_falsey_until_something_is_asked_for() -> None:
    expansion = Expansion()
    assert not expansion

    expansion.add(build_node("x", "fake"))
    assert expansion


# --- depth ------------------------------------------------------------------------------


def test_a_node_present_at_the_start_is_generation_zero() -> None:
    graph = graph_for(max_depth=1)

    assert graph.apply("root", asking_for(build_node("child", "fake"))) != ()


def test_what_an_expansion_added_may_not_expand_again_by_default() -> None:
    # Depth 1 by default: a planner may plan, and what it plans may not plan further.
    graph = graph_for(max_depth=1)
    graph.apply("root", asking_for(build_node("child", "fake")))

    with pytest.raises(ValidationError, match="depth 2, above the limit of 1"):
        graph.apply("child", asking_for(build_node("grandchild", "fake")))


def test_raising_the_depth_limit_allows_another_generation() -> None:
    graph = graph_for(max_depth=2)
    graph.apply("root", asking_for(build_node("child", "fake")))

    added = graph.apply("child", asking_for(build_node("grandchild", "fake")))

    assert [node.id for node in added] == ["grandchild"]


def test_depth_zero_forbids_expansion_outright() -> None:
    graph = graph_for(max_depth=0)

    with pytest.raises(ValidationError, match="above the limit of 0"):
        graph.apply("root", asking_for(build_node("child", "fake")))


def test_a_rejected_expansion_leaves_the_graph_alone() -> None:
    graph = graph_for(max_depth=0)
    before = graph.workflow

    with pytest.raises(ValidationError):
        graph.apply("root", asking_for(build_node("child", "fake")))

    assert graph.workflow is before


def test_an_unknown_source_counts_as_generation_zero() -> None:
    # A node the graph has never heard of cannot be trusted to have a depth, and the
    # conservative reading — treat it as a root — is the one that still enforces a limit.
    graph = graph_for(max_depth=1)

    assert graph.apply("who?", asking_for(build_node("child", "fake"))) != ()


# --- size --------------------------------------------------------------------------------


def test_the_node_ceiling_stops_a_runaway_expansion() -> None:
    graph = graph_for(max_nodes=3)

    with pytest.raises(ValidationError, match="above the limit of 3"):
        graph.apply(
            "root",
            asking_for(*(build_node(f"n{index}", "fake") for index in range(5))),
        )


def test_an_expansion_that_fits_under_the_ceiling_is_allowed() -> None:
    graph = graph_for(max_nodes=3)

    added = graph.apply("root", asking_for(build_node("a", "fake"), build_node("b", "fake")))

    assert len(added) == 2


def test_the_ceiling_counts_the_whole_graph_not_just_the_expansion() -> None:
    # Which is what makes it the bound that survives a crash: it is measured against the
    # persisted graph, where the in-memory depth counter is not.
    graph = graph_for(max_nodes=2)
    graph.apply("root", asking_for(build_node("a", "fake")))

    with pytest.raises(ValidationError, match="above the limit of 2"):
        graph.apply("root", asking_for(build_node("b", "fake")))


# --- rejection ---------------------------------------------------------------------------


def test_an_expansion_that_would_create_a_cycle_is_rejected() -> None:
    graph = graph_for()

    with pytest.raises(ValidationError) as caught:
        graph.apply(
            "root",
            asking_for(
                build_node("x", "fake", depends_on=["y"]),
                build_node("y", "fake", depends_on=["x"]),
            ),
        )

    assert caught.value.cycle is not None


def test_an_unregistered_agent_is_rejected() -> None:
    graph = graph_for()

    with pytest.raises(ValidationError, match="not registered"):
        graph.apply("root", asking_for(build_node("child", "nobody")))


def test_redefining_an_existing_node_is_rejected() -> None:
    graph = graph_for()

    with pytest.raises(ValidationError, match="redefines node 'root'"):
        graph.apply("root", asking_for(build_node("root", "fake", params={"changed": True})))
