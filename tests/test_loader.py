import pathlib
import re

import pytest

from dagent.errors import DagentError, ValidationError
from dagent.loader import load_workflow, load_workflow_file
from dagent.models.workflow import Policy

DIAMOND = """
name: diamond
nodes:
  - id: a
    agent: fake
  - id: b
    agent: fake
    inputs:
      x: a
  - id: c
    agent: fake
    inputs:
      x: a
  - id: d
    agent: fake
    inputs:
      left: b
      right: c
"""


def test_a_workflow_round_trips_from_yaml() -> None:
    workflow = load_workflow(DIAMOND)

    assert workflow.name == "diamond"
    assert [node.id for node in workflow.nodes] == ["a", "b", "c", "d"]


def test_inputs_imply_edges_exactly_as_the_builder_does() -> None:
    # The loader builds through WorkflowBuilder, so files get no separate rulebook.
    workflow = load_workflow(DIAMOND)

    assert workflow.nodes[3].depends_on == ("b", "c")


def test_a_loaded_workflow_is_frozen() -> None:
    workflow = load_workflow(DIAMOND)

    assert isinstance(workflow.nodes, tuple)


def test_ordering_only_edges_load() -> None:
    workflow = load_workflow(
        "name: w\nnodes:\n  - id: a\n    agent: fake\n  - id: b\n    agent: fake\n"
        "    depends_on: [a]\n"
    )

    assert workflow.nodes[1].depends_on == ("a",)


def test_params_load() -> None:
    workflow = load_workflow(
        "name: w\nnodes:\n  - id: a\n    agent: constant\n    params:\n      value: hello\n"
    )

    assert dict(workflow.nodes[0].params) == {"value": "hello"}


def test_params_accept_structured_values() -> None:
    workflow = load_workflow(
        "name: w\nnodes:\n  - id: a\n    agent: constant\n"
        "    params:\n      value:\n        - one\n        - two\n"
    )

    assert dict(workflow.nodes[0].params) == {"value": ["one", "two"]}


def test_a_policy_override_loads() -> None:
    workflow = load_workflow(
        "name: w\nnodes:\n  - id: a\n    agent: fake\n"
        "    policy:\n      max_attempts: 3\n      timeout_s: 30\n"
    )

    assert workflow.nodes[0].policy == Policy(max_attempts=3, timeout_s=30)


# --- rejections ---------------------------------------------------------------------


def test_malformed_yaml_is_rejected() -> None:
    with pytest.raises(ValidationError, match="invalid YAML"):
        load_workflow("name: [unclosed")


def test_a_non_mapping_document_is_rejected() -> None:
    with pytest.raises(ValidationError, match="mapping at the top level"):
        load_workflow("- just\n- a list\n")


def test_a_missing_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="'name' is required"):
        load_workflow("nodes: []")


def test_missing_nodes_is_rejected() -> None:
    with pytest.raises(ValidationError, match="'nodes' is required"):
        load_workflow("name: w")


def test_an_unknown_top_level_key_is_rejected() -> None:
    # A typo'd key should fail loudly, not be silently ignored (AGENTS.md rule 6).
    with pytest.raises(ValidationError, match="unknown top-level key"):
        load_workflow("name: w\nnodes: []\nnods: []\n")


def test_an_unknown_node_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown key"):
        load_workflow("name: w\nnodes:\n  - id: a\n    agent: fake\n    dependson: []\n")


def test_a_node_without_an_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="'id' is required"):
        load_workflow("name: w\nnodes:\n  - agent: fake\n")


def test_a_node_without_an_agent_is_rejected() -> None:
    with pytest.raises(ValidationError, match="'agent' is required"):
        load_workflow("name: w\nnodes:\n  - id: a\n")


def test_a_non_mapping_node_is_rejected() -> None:
    with pytest.raises(ValidationError, match="nodes\\[0\\]"):
        load_workflow("name: w\nnodes:\n  - just a string\n")


def test_bad_depends_on_is_rejected() -> None:
    with pytest.raises(ValidationError, match="'depends_on'"):
        load_workflow("name: w\nnodes:\n  - id: a\n    agent: fake\n    depends_on: nope\n")


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValidationError, match="'inputs'"):
        load_workflow("name: w\nnodes:\n  - id: a\n    agent: fake\n    inputs: nope\n")


def test_bad_params_are_rejected() -> None:
    with pytest.raises(ValidationError, match="'params'"):
        load_workflow("name: w\nnodes:\n  - id: a\n    agent: constant\n    params: nope\n")


def test_a_non_mapping_policy_is_rejected() -> None:
    with pytest.raises(ValidationError, match="'policy' must be a mapping"):
        load_workflow("name: w\nnodes:\n  - id: a\n    agent: fake\n    policy: nope\n")


def test_an_invalid_policy_is_rejected() -> None:
    with pytest.raises(ValidationError, match="invalid policy"):
        load_workflow(
            "name: w\nnodes:\n  - id: a\n    agent: fake\n    policy:\n      max_attempts: 0\n"
        )


def test_graph_rules_still_apply_to_loaded_files() -> None:
    with pytest.raises(ValidationError) as raised:
        load_workflow(
            "name: w\nnodes:\n  - id: a\n    agent: fake\n    depends_on: [b]\n"
            "  - id: b\n    agent: fake\n    depends_on: [a]\n"
        )

    assert raised.value.cycle is not None


def test_the_source_name_appears_in_errors() -> None:
    with pytest.raises(ValidationError, match=re.escape("my-file.yaml")):
        load_workflow("nodes: []", source="my-file.yaml")


def test_every_rejection_is_catchable_as_a_dagent_error() -> None:
    with pytest.raises(DagentError):
        load_workflow("name: [unclosed")


# --- file access --------------------------------------------------------------------


def test_a_file_loads(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "w.yaml"
    path.write_text(DIAMOND, encoding="utf-8")

    assert load_workflow_file(path).name == "diamond"


def test_a_missing_file_is_a_validation_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValidationError, match="cannot read workflow file"):
        load_workflow_file(tmp_path / "nope.yaml")


def test_the_file_path_appears_in_parse_errors(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("nodes: []", encoding="utf-8")

    with pytest.raises(ValidationError, match=re.escape("broken.yaml")):
        load_workflow_file(path)


def test_the_shipped_research_example_loads_and_validates() -> None:
    example = pathlib.Path(__file__).parent.parent / "examples" / "research.yaml"

    workflow = load_workflow_file(example)

    assert workflow.name == "research"
    assert {node.agent for node in workflow.nodes} == {"constant", "researcher", "synthesizer"}


# --- the shipped YAML examples --------------------------------------------------------

YAML_EXAMPLES = sorted((pathlib.Path(__file__).parent.parent / "examples").glob("*.yaml"))


def test_there_are_yaml_examples_to_check() -> None:
    assert YAML_EXAMPLES


@pytest.mark.parametrize("path", YAML_EXAMPLES, ids=lambda p: p.name)
def test_every_shipped_example_loads_and_validates(path: pathlib.Path) -> None:
    # An example that does not load is documentation that lies. The agent check runs too,
    # so an example naming an agent nobody registered fails here rather than at a demo.
    import dagent.agents  # noqa: F401  — registers the built-ins
    from dagent.graph.validate import validate
    from dagent.runtime.registry import default_registry

    workflow = load_workflow_file(path)

    validate(workflow, known_agents=default_registry.names())


@pytest.mark.asyncio
async def test_the_dynamic_example_grows_its_own_graph() -> None:
    # Two nodes written down; more than two nodes after it runs. That difference is FR-7.
    import dagent.agents  # noqa: F401
    from dagent.runtime.executor import Executor
    from dagent.runtime.model import StubModelClient
    from dagent.runtime.registry import default_registry
    from dagent.store.memory import InMemoryStateStore

    path = pathlib.Path(__file__).parent.parent / "examples" / "research_dynamic.yaml"
    workflow = load_workflow_file(path)
    store = InMemoryStateStore()

    run = await Executor(
        registry=default_registry,
        store=store,
        model=StubModelClient(lambda request: "first topic\nsecond topic\nthird topic"),
    ).run(workflow, run_id="dyn")

    assert len(workflow.nodes) == 2
    assert sorted(run.nodes) == [
        "plan",
        "plan.research_0",
        "plan.research_1",
        "plan.research_2",
        "plan.synthesis",
        "question",
    ]
    assert run.state.value == "succeeded"
