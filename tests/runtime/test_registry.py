import pytest

from dagent.errors import AgentError, DagentError
from dagent.models.state import NodeOutput
from dagent.runtime.agent import AgentContext
from dagent.runtime.registry import AgentRegistry, default_registry, register


class Stub:
    async def run(self, ctx: AgentContext) -> NodeOutput:
        return {"node": ctx.node_id}


def test_a_registered_agent_can_be_created() -> None:
    registry = AgentRegistry()
    registry.add("stub", Stub)

    assert isinstance(registry.create("stub"), Stub)


def test_each_call_builds_a_fresh_agent() -> None:
    # Factories, not singletons: one node's state must not leak into another's.
    registry = AgentRegistry()
    registry.add("stub", Stub)

    assert registry.create("stub") is not registry.create("stub")


def test_the_decorator_registers_and_returns_the_class_unchanged() -> None:
    registry = AgentRegistry()

    @registry.register("stub")
    class Decorated:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            return None

    assert "stub" in registry
    assert isinstance(registry.create("stub"), Decorated)


def test_names_are_what_graph_validation_gets_injected() -> None:
    registry = AgentRegistry()
    registry.add("one", Stub)
    registry.add("two", Stub)

    assert registry.names() == frozenset({"one", "two"})


def test_an_empty_registry_reports_no_names() -> None:
    assert AgentRegistry().names() == frozenset()
    assert len(AgentRegistry()) == 0


def test_registering_a_duplicate_name_is_refused() -> None:
    # Silently replacing would make a workflow's behaviour depend on import order.
    registry = AgentRegistry()
    registry.add("stub", Stub)

    with pytest.raises(AgentError, match="already registered"):
        registry.add("stub", Stub)


def test_an_empty_name_is_refused() -> None:
    with pytest.raises(AgentError):
        AgentRegistry().add("", Stub)


def test_creating_an_unregistered_agent_raises() -> None:
    with pytest.raises(AgentError, match="no agent registered"):
        AgentRegistry().create("ghost")


def test_registry_failures_are_catchable_as_dagent_errors() -> None:
    with pytest.raises(DagentError):
        AgentRegistry().create("ghost")


def test_registries_are_independent() -> None:
    first, second = AgentRegistry(), AgentRegistry()
    first.add("stub", Stub)

    assert "stub" not in second


def test_the_module_level_decorator_writes_to_the_default_registry() -> None:
    name = "test-default-registry-agent"

    @register(name)
    class Global:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            return None

    try:
        assert name in default_registry
        assert isinstance(default_registry.create(name), Global)
    finally:
        # Reaching into the private rather than adding an unregister() the engine does
        # not need: AGENTS.md rule 2 forbids widening a public interface for a test.
        default_registry._factories.pop(name, None)
