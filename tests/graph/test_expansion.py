"""The pure merge: appending nodes to a workflow and revalidating the result.

Every rule that needs run state — how deep, how big, who asked — is tested in
`tests/runtime/test_run_graph.py` against `RunGraph`. What is here is total and pure and
needs no executor.
"""

import pytest

from dagent.errors import ValidationError
from dagent.graph.builder import build_node
from dagent.graph.expansion import expand_workflow
from dagent.models.workflow import Workflow

KNOWN = {"fake"}


def test_expansion_appends_the_new_nodes(diamond: Workflow) -> None:
    augmented = expand_workflow(
        diamond, [build_node("e", "fake", inputs={"x": "d"})], known_agents=KNOWN
    )

    assert [node.id for node in augmented.nodes] == ["a", "b", "c", "d", "e"]


def test_expansion_preserves_the_existing_nodes_untouched(diamond: Workflow) -> None:
    # Append-only is the guardrail, not an implementation detail: a node already RUNNING
    # or SUCCESS never re-checks its dependencies, so it must never gain one.
    augmented = expand_workflow(diamond, [build_node("e", "fake")], known_agents=KNOWN)

    assert augmented.nodes[: len(diamond.nodes)] == diamond.nodes


def test_the_original_workflow_is_not_mutated(diamond: Workflow) -> None:
    before = diamond.nodes

    expand_workflow(diamond, [build_node("e", "fake")], known_agents=KNOWN)

    assert diamond.nodes == before


def test_expanding_with_nothing_returns_the_same_workflow(diamond: Workflow) -> None:
    assert expand_workflow(diamond, [], known_agents=KNOWN) is diamond


def test_an_input_gets_its_edge_without_being_asked(diamond: Workflow) -> None:
    # build_node does for a planner exactly what WorkflowBuilder does for an author, so a
    # generated node cannot accidentally read from a node it does not wait for.
    augmented = expand_workflow(
        diamond, [build_node("e", "fake", inputs={"from_d": "d"})], known_agents=KNOWN
    )

    assert augmented.nodes[-1].depends_on == ("d",)


def test_a_new_node_may_depend_on_a_node_that_has_already_finished(diamond: Workflow) -> None:
    # The showcase rests on this: a planner's children depend on the planner, which has
    # necessarily finished by the time it asks for them. Its output is in the store, so
    # they are ready immediately — there is nothing to strand.
    augmented = expand_workflow(
        diamond, [build_node("e", "fake", inputs={"x": "a"})], known_agents=KNOWN
    )

    assert augmented.nodes[-1].depends_on == ("a",)


def test_new_nodes_may_depend_on_one_another(diamond: Workflow) -> None:
    added = [
        build_node("e", "fake", inputs={"x": "d"}),
        build_node("f", "fake", inputs={"y": "e"}),
    ]

    augmented = expand_workflow(diamond, added, known_agents=KNOWN)

    assert augmented.nodes[-1].depends_on == ("e",)


# --- what it refuses ------------------------------------------------------------------


def test_redefining_an_existing_node_is_refused(diamond: Workflow) -> None:
    with pytest.raises(ValidationError, match="redefines node 'a'"):
        expand_workflow(diamond, [build_node("a", "fake", params={"new": 1})], known_agents=KNOWN)


def test_restating_an_existing_node_identically_is_a_no_op(diamond: Workflow) -> None:
    # A planner re-executed after a crash replays its own expansion. Treating that as a
    # collision would make the very thing Phase 5 guarantees — safe re-execution —
    # impossible for a planner (DR-4).
    augmented = expand_workflow(diamond, list(diamond.nodes), known_agents=KNOWN)

    assert augmented is diamond


def test_the_same_new_node_twice_in_one_request_is_added_once(diamond: Workflow) -> None:
    node = build_node("e", "fake")

    augmented = expand_workflow(diamond, [node, node], known_agents=KNOWN)

    assert [n.id for n in augmented.nodes].count("e") == 1


def test_two_different_nodes_under_one_id_are_refused(diamond: Workflow) -> None:
    added = [build_node("e", "fake"), build_node("e", "fake", params={"different": True})]

    with pytest.raises(ValidationError, match="redefines node 'e'"):
        expand_workflow(diamond, added, known_agents=KNOWN)


def test_an_unregistered_agent_is_refused(diamond: Workflow) -> None:
    with pytest.raises(ValidationError, match="not registered"):
        expand_workflow(diamond, [build_node("e", "nobody")], known_agents=KNOWN)


def test_a_reference_to_a_node_that_does_not_exist_is_refused(diamond: Workflow) -> None:
    with pytest.raises(ValidationError, match="ghost"):
        expand_workflow(
            diamond, [build_node("e", "fake", inputs={"x": "ghost"})], known_agents=KNOWN
        )


def test_a_cycle_among_the_new_nodes_is_refused(diamond: Workflow) -> None:
    added = [
        build_node("e", "fake", depends_on=["f"]),
        build_node("f", "fake", depends_on=["e"]),
    ]

    with pytest.raises(ValidationError) as caught:
        expand_workflow(diamond, added, known_agents=KNOWN)

    assert caught.value.cycle is not None


def test_a_self_referential_new_node_is_refused(diamond: Workflow) -> None:
    # The validator runs over the whole augmented graph, not just over what is new.
    with pytest.raises(ValidationError) as caught:
        expand_workflow(diamond, [build_node("e", "fake", depends_on=["e"])], known_agents=KNOWN)

    assert caught.value.cycle == ("e", "e")


def test_a_rejected_expansion_leaves_the_caller_holding_a_valid_workflow(
    diamond: Workflow,
) -> None:
    with pytest.raises(ValidationError):
        expand_workflow(diamond, [build_node("e", "nobody")], known_agents=KNOWN)

    assert [node.id for node in diamond.nodes] == ["a", "b", "c", "d"]


def test_the_agent_check_is_skipped_when_no_registry_is_supplied(diamond: Workflow) -> None:
    augmented = expand_workflow(diamond, [build_node("e", "not_registered_anywhere")])

    assert augmented.nodes[-1].agent == "not_registered_anywhere"
