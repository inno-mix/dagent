"""Test graph expansion and validation logic (Phase 6, Task 2)."""

import pytest

from dagent.errors import ValidationError
from dagent.graph.expansion import expand_workflow, validate_expansion
from dagent.models.expansion import NodeDefinition
from dagent.models.workflow import Node, Workflow


class TestExpandWorkflow:
    """Test the low-level expand_workflow function."""

    def test_expand_workflow_adds_new_nodes(self, diamond: Workflow) -> None:
        """expand_workflow should add new nodes to an existing workflow."""
        new_nodes = [
            NodeDefinition(
                name="e",
                agent="fake",
                inputs={"x": "d"},
            ),
        ]
        expanded = expand_workflow(diamond, new_nodes, known_agents={"fake"})

        assert len(expanded.nodes) == 5  # 4 original + 1 new
        assert expanded.nodes[-1].id == "e"
        assert expanded.nodes[-1].agent == "fake"
        assert expanded.nodes[-1].inputs == {"x": "d"}

    def test_expand_workflow_preserves_existing_nodes(self, diamond: Workflow) -> None:
        """expand_workflow should not mutate existing nodes."""
        new_nodes = [NodeDefinition(name="e", agent="fake")]
        expanded = expand_workflow(diamond, new_nodes, known_agents={"fake"})

        # Check original nodes are unchanged
        for i, original_node in enumerate(diamond.nodes):
            expanded_original = expanded.nodes[i]
            assert expanded_original.id == original_node.id
            assert expanded_original.agent == original_node.agent
            assert expanded_original.depends_on == original_node.depends_on
            assert expanded_original.inputs == original_node.inputs

    def test_expand_workflow_rejects_name_collision(self, diamond: Workflow) -> None:
        """expand_workflow should reject new nodes with names that collide with existing."""
        new_nodes = [NodeDefinition(name="a", agent="fake")]  # 'a' already exists

        with pytest.raises(ValidationError, match="already exists|collision|duplicate"):
            expand_workflow(diamond, new_nodes, known_agents={"fake"})

    def test_expand_workflow_rejects_unknown_agent(self, diamond: Workflow) -> None:
        """expand_workflow should reject new nodes with unregistered agents."""
        new_nodes = [NodeDefinition(name="e", agent="unknown")]

        with pytest.raises(ValidationError, match="not registered|unknown"):
            expand_workflow(diamond, new_nodes, known_agents={"fake"})

    def test_expand_workflow_calls_validate(self, diamond: Workflow) -> None:
        """expand_workflow should validate the augmented graph."""
        new_nodes = [
            NodeDefinition(name="e", agent="fake", inputs={"x": "ghost"})
        ]

        with pytest.raises(ValidationError, match="ghost|not in this workflow"):
            expand_workflow(diamond, new_nodes, known_agents={"fake"})

    def test_expand_workflow_rejects_cycle_in_augmented_graph(
        self, diamond: Workflow
    ) -> None:
        """expand_workflow should reject cycles introduced by new nodes."""
        # Try to create a cycle: d depends on new node e, which depends on d
        new_nodes = [
            NodeDefinition(
                name="e",
                agent="fake",
                inputs={"x": "d"},
                params={},
            ),
            NodeDefinition(
                name="f",
                agent="fake",
                inputs={"x": "e"},
                params={},
            ),
            # Now make d depend on f (which will be added), creating a cycle
        ]
        # Actually, we can't easily create a cycle here because d already exists.
        # Let's test differently: add e depending on d, then f depending on e,
        # then try to make d depend on f (but we can't modify d).
        # This test documents the limit: expand_workflow can only validate cycles
        # created by the NEW nodes depending on OLD ones, not the reverse.
        # For now, we'll test that cycles between new nodes are detected.

        new_nodes = [
            NodeDefinition(name="e", agent="fake", inputs={"x": "f"}),
            NodeDefinition(name="f", agent="fake", inputs={"x": "e"}),
        ]

        with pytest.raises(ValidationError, match="cycle"):
            expand_workflow(diamond, new_nodes, known_agents={"fake"})

    def test_expand_workflow_with_empty_new_nodes(self, diamond: Workflow) -> None:
        """expand_workflow should handle empty new_nodes gracefully."""
        expanded = expand_workflow(diamond, [], known_agents={"fake"})

        assert expanded.nodes == diamond.nodes
        assert len(expanded.nodes) == 4

    def test_expand_workflow_adds_depends_on_edge_for_inputs(
        self, diamond: Workflow
    ) -> None:
        """expand_workflow should add depends_on edges for all referenced inputs."""
        new_nodes = [
            NodeDefinition(
                name="e",
                agent="fake",
                inputs={"x": "b", "y": "c"},
            ),
        ]
        expanded = expand_workflow(diamond, new_nodes, known_agents={"fake"})

        e_node = expanded.nodes[-1]
        assert "b" in e_node.depends_on
        assert "c" in e_node.depends_on


class TestValidateExpansion:
    """Test the pre-expansion validation guard."""

    def test_validate_expansion_accepts_valid_expansion(
        self, diamond: Workflow
    ) -> None:
        """validate_expansion should accept valid expansions."""
        new_nodes = [
            NodeDefinition(
                name="e",
                agent="fake",
                inputs={"x": "d"},
            ),
        ]

        # Should not raise: 'd' is running, so we can depend on it
        validate_expansion(
            diamond,
            new_nodes,
            running_nodes={"d"},
            expansion_count=1,
            max_depth=100,
            known_agents={"fake"},
        )

    def test_validate_expansion_rejects_depth_exceeded(
        self, diamond: Workflow
    ) -> None:
        """validate_expansion should reject when expansion_count >= max_depth."""
        new_nodes = [NodeDefinition(name="e", agent="fake")]

        with pytest.raises(ValidationError, match="depth|exceeded|max_depth"):
            validate_expansion(
                diamond,
                new_nodes,
                running_nodes=set(),
                expansion_count=100,  # Already at limit
                max_depth=100,
                known_agents={"fake"},
            )

    def test_validate_expansion_strand_prevention(self, diamond: Workflow) -> None:
        """validate_expansion should reject nodes depending on already-SUCCESS nodes."""
        # Simulate that 'a' is already SUCCESS (not in running_nodes)
        new_nodes = [
            NodeDefinition(
                name="e",
                agent="fake",
                inputs={"x": "a"},  # 'a' is SUCCESS, not running
            ),
        ]

        with pytest.raises(ValidationError, match="strand|SUCCESS|already"):
            validate_expansion(
                diamond,
                new_nodes,
                running_nodes={"b", "c", "d"},  # 'a' is SUCCESS
                expansion_count=0,
                max_depth=100,
                known_agents={"fake"},
            )

    def test_validate_expansion_allows_depending_on_running_nodes(
        self, diamond: Workflow
    ) -> None:
        """validate_expansion should allow new nodes to depend on running nodes."""
        new_nodes = [
            NodeDefinition(
                name="e",
                agent="fake",
                inputs={"x": "b"},  # 'b' is running
            ),
        ]

        # Should not raise
        validate_expansion(
            diamond,
            new_nodes,
            running_nodes={"b", "c", "d"},  # 'b' is running
            expansion_count=0,
            max_depth=100,
            known_agents={"fake"},
        )

    def test_validate_expansion_allows_depending_on_other_new_nodes(
        self, diamond: Workflow
    ) -> None:
        """validate_expansion should allow new nodes to depend on other new nodes."""
        new_nodes = [
            NodeDefinition(name="e", agent="fake", inputs={"x": "d"}),
            NodeDefinition(name="f", agent="fake", inputs={"x": "e"}),
        ]

        # Should not raise: 'd' is running, and 'f' can depend on 'e' (another new node)
        validate_expansion(
            diamond,
            new_nodes,
            running_nodes={"d"},  # 'd' is still running
            expansion_count=1,
            max_depth=100,
            known_agents={"fake"},
        )

    def test_validate_expansion_rejects_unknown_agent(self, diamond: Workflow) -> None:
        """validate_expansion should reject unknown agents."""
        new_nodes = [NodeDefinition(name="e", agent="unknown")]

        with pytest.raises(ValidationError, match="not registered|unknown"):
            validate_expansion(
                diamond,
                new_nodes,
                running_nodes=set(),
                expansion_count=0,
                max_depth=100,
                known_agents={"fake"},
            )

    def test_validate_expansion_rejects_reference_to_nonexistent_node(
        self, diamond: Workflow
    ) -> None:
        """validate_expansion should reject inputs referencing nonexistent nodes."""
        new_nodes = [
            NodeDefinition(
                name="e",
                agent="fake",
                inputs={"x": "ghost"},
            ),
        ]

        with pytest.raises(ValidationError, match="ghost|not in workflow|does not exist"):
            validate_expansion(
                diamond,
                new_nodes,
                running_nodes=set(),
                expansion_count=0,
                max_depth=100,
                known_agents={"fake"},
            )

    def test_validate_expansion_respects_max_depth_boundary(
        self, diamond: Workflow
    ) -> None:
        """validate_expansion should allow expansion_count < max_depth."""
        new_nodes = [NodeDefinition(name="e", agent="fake")]

        # At max_depth - 1 should work
        validate_expansion(
            diamond,
            new_nodes,
            running_nodes=set(),
            expansion_count=99,
            max_depth=100,
            known_agents={"fake"},
        )

    def test_validate_expansion_rejects_at_max_depth(
        self, diamond: Workflow
    ) -> None:
        """validate_expansion should reject when expansion_count == max_depth."""
        new_nodes = [NodeDefinition(name="e", agent="fake")]

        with pytest.raises(ValidationError):
            validate_expansion(
                diamond,
                new_nodes,
                running_nodes=set(),
                expansion_count=100,
                max_depth=100,
                known_agents={"fake"},
            )

    def test_validate_expansion_multiple_inputs_strand_prevention(
        self, diamond: Workflow
    ) -> None:
        """validate_expansion should reject if ANY input is to a SUCCESS node."""
        new_nodes = [
            NodeDefinition(
                name="e",
                agent="fake",
                inputs={"x": "d", "y": "a"},  # 'a' is SUCCESS
            ),
        ]

        with pytest.raises(ValidationError, match="strand|SUCCESS"):
            validate_expansion(
                diamond,
                new_nodes,
                running_nodes={"b", "c", "d"},  # 'a' is SUCCESS
                expansion_count=0,
                max_depth=100,
                known_agents={"fake"},
            )

    def test_validate_expansion_with_params_no_validation_error(
        self, diamond: Workflow
    ) -> None:
        """validate_expansion should handle params without validation errors."""
        new_nodes = [
            NodeDefinition(
                name="e",
                agent="fake",
                params={"threshold": 0.5, "topic": "test"},
            ),
        ]

        # Should not raise
        validate_expansion(
            diamond,
            new_nodes,
            running_nodes=set(),
            expansion_count=0,
            max_depth=100,
            known_agents={"fake"},
        )
