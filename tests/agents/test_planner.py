"""Tests for node expansion schema used by planner agents."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from dagent.models.expansion import ExpansionResult, NodeDefinition
from dagent.models.workflow import Policy


class TestNodeDefinition:
    """Tests for NodeDefinition model."""

    def test_node_definition_with_minimal_fields(self) -> None:
        """Create a NodeDefinition with only required fields."""
        node = NodeDefinition(
            name="my_node",
            agent="constant",
            inputs={},
            params={},
        )
        assert node.name == "my_node"
        assert node.agent == "constant"
        assert node.inputs == {}
        assert node.params == {}
        assert node.policy is None

    def test_node_definition_with_all_fields(self) -> None:
        """Create a NodeDefinition with all fields including policy."""
        policy = Policy(max_attempts=3, timeout_s=60.0)
        node = NodeDefinition(
            name="research_node",
            agent="researcher",
            inputs={"topic": "seed_node"},
            params={"depth": 3, "style": "technical"},
            policy=policy,
        )
        assert node.name == "research_node"
        assert node.agent == "researcher"
        assert node.inputs == {"topic": "seed_node"}
        assert node.params == {"depth": 3, "style": "technical"}
        assert node.policy is not None
        assert node.policy.max_attempts == 3

    def test_node_definition_with_complex_params(self) -> None:
        """NodeDefinition accepts nested JSON structures in params."""
        node = NodeDefinition(
            name="complex_node",
            agent="processor",
            inputs={},
            params={
                "simple": "value",
                "number": 42,
                "float_val": 3.14,
                "bool_val": True,
                "none_val": None,
                "list_val": [1, 2, 3],
                "nested": {"key": "value", "count": 10},
            },
        )
        assert node.params["simple"] == "value"
        assert node.params["number"] == 42
        assert node.params["list_val"] == [1, 2, 3]
        assert node.params["nested"]["key"] == "value"

    def test_node_definition_is_frozen(self) -> None:
        """NodeDefinition is immutable (frozen)."""
        node = NodeDefinition(
            name="immutable_node",
            agent="constant",
            inputs={},
            params={},
        )
        with pytest.raises(PydanticValidationError):
            node.name = "changed"  # type: ignore[misc]

    def test_node_definition_serialization(self) -> None:
        """NodeDefinition can be serialized to JSON-compatible dict."""
        policy = Policy(max_attempts=2)
        node = NodeDefinition(
            name="test_node",
            agent="test_agent",
            inputs={"source": "upstream"},
            params={"key": "value"},
            policy=policy,
        )
        data = node.model_dump()
        assert data["name"] == "test_node"
        assert data["agent"] == "test_agent"
        assert data["inputs"] == {"source": "upstream"}
        assert data["params"] == {"key": "value"}
        assert data["policy"]["max_attempts"] == 2

    def test_node_definition_deserialization(self) -> None:
        """NodeDefinition can be created from JSON-compatible dict."""
        data = {
            "name": "reconstructed",
            "agent": "fake",
            "inputs": {"a": "b"},
            "params": {"x": 1},
            "policy": {"max_attempts": 5, "timeout_s": 10.0},
        }
        node = NodeDefinition(**data)
        assert node.name == "reconstructed"
        assert node.agent == "fake"
        assert node.policy.max_attempts == 5


class TestExpansionResult:
    """Tests for ExpansionResult model."""

    def test_expansion_result_with_single_node(self) -> None:
        """Create an ExpansionResult with a single node."""
        node = NodeDefinition(
            name="node1",
            agent="constant",
            inputs={},
            params={},
        )
        result = ExpansionResult(nodes=[node])
        assert len(result.nodes) == 1
        assert result.nodes[0].name == "node1"
        assert result.max_depth == 100  # default

    def test_expansion_result_with_multiple_nodes(self) -> None:
        """Create an ExpansionResult with multiple nodes."""
        nodes = [
            NodeDefinition(
                name="node1",
                agent="constant",
                inputs={},
                params={"value": "seed"},
            ),
            NodeDefinition(
                name="node2",
                agent="researcher",
                inputs={"topic": "node1"},
                params={},
            ),
            NodeDefinition(
                name="node3",
                agent="synthesizer",
                inputs={"research": "node2"},
                params={},
            ),
        ]
        result = ExpansionResult(nodes=nodes)
        assert len(result.nodes) == 3
        assert result.nodes[0].name == "node1"
        assert result.nodes[1].inputs["topic"] == "node1"
        assert result.nodes[2].inputs["research"] == "node2"

    def test_expansion_result_with_custom_max_depth(self) -> None:
        """Create an ExpansionResult with custom max_depth."""
        result = ExpansionResult(nodes=[], max_depth=50)
        assert result.max_depth == 50

    def test_expansion_result_max_depth_range(self) -> None:
        """max_depth must be between 1 and 1000."""
        # Valid range
        result = ExpansionResult(nodes=[], max_depth=1)
        assert result.max_depth == 1

        result = ExpansionResult(nodes=[], max_depth=1000)
        assert result.max_depth == 1000

        # Invalid: too low
        with pytest.raises(PydanticValidationError):
            ExpansionResult(nodes=[], max_depth=0)

        # Invalid: too high
        with pytest.raises(PydanticValidationError):
            ExpansionResult(nodes=[], max_depth=1001)

    def test_expansion_result_is_frozen(self) -> None:
        """ExpansionResult is immutable (frozen)."""
        result = ExpansionResult(nodes=[])
        with pytest.raises(PydanticValidationError):
            result.max_depth = 200  # type: ignore[misc]

    def test_expansion_result_serialization(self) -> None:
        """ExpansionResult can be serialized to JSON-compatible dict."""
        nodes = [
            NodeDefinition(
                name="n1",
                agent="a1",
                inputs={},
                params={},
            ),
        ]
        result = ExpansionResult(nodes=nodes, max_depth=75)
        data = result.model_dump()
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["name"] == "n1"
        assert data["max_depth"] == 75

    def test_expansion_result_deserialization(self) -> None:
        """ExpansionResult can be created from JSON-compatible dict."""
        data = {
            "nodes": [
                {
                    "name": "node1",
                    "agent": "agent1",
                    "inputs": {},
                    "params": {"x": 1},
                }
            ],
            "max_depth": 200,
        }
        result = ExpansionResult(**data)
        assert len(result.nodes) == 1
        assert result.nodes[0].name == "node1"
        assert result.max_depth == 200

    def test_expansion_result_empty_nodes_list(self) -> None:
        """ExpansionResult can have an empty nodes list."""
        result = ExpansionResult(nodes=[])
        assert result.nodes == []
        assert result.max_depth == 100
