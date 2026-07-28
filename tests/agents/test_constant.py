import pytest

from dagent.agents.fake import ConstantAgent
from dagent.errors import AgentError
from dagent.graph.builder import WorkflowBuilder
from dagent.models.state import RunState
from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import ManualClock
from dagent.runtime.executor import Executor
from dagent.runtime.registry import default_registry
from dagent.store.memory import InMemoryStateStore


def a_context(node_id: str = "seed", **params: object) -> AgentContext:
    return AgentContext(
        run_id="r1",
        node_id=node_id,
        attempt=0,
        inputs={},
        clock=ManualClock(),
        params=dict(params),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_the_constant_agent_emits_its_value() -> None:
    assert await ConstantAgent().run(a_context(value="a topic")) == "a topic"


@pytest.mark.asyncio
async def test_the_constant_agent_emits_structured_values_unchanged() -> None:
    payload = {"nested": [1, 2, 3]}

    assert await ConstantAgent().run(a_context(value=payload)) == payload


@pytest.mark.asyncio
async def test_the_constant_agent_can_emit_none() -> None:
    # None is a legitimate output, distinct from "no value parameter".
    assert await ConstantAgent().run(a_context(value=None)) is None


@pytest.mark.asyncio
async def test_the_constant_agent_says_what_is_missing() -> None:
    with pytest.raises(AgentError, match="needs a 'value' parameter"):
        await ConstantAgent().run(a_context(wrong_key="x"))


@pytest.mark.asyncio
async def test_params_reach_the_agent_through_the_executor() -> None:
    # The whole point of params: a workflow needs a source, and this is it.
    store = InMemoryStateStore()
    workflow = (
        WorkflowBuilder("seeded")
        .add_node("seed", "constant", params={"value": "the seed value"})
        .add_node("downstream", "fake", inputs={"from_seed": "seed"})
        .build(known_agents=default_registry.names())
    )

    run = await Executor(registry=default_registry, store=store).run(workflow, run_id="r1")

    assert run.state is RunState.SUCCEEDED
    assert await store.load_output("r1", "seed") == "the seed value"
    downstream = await store.load_output("r1", "downstream")
    assert isinstance(downstream, dict)
    assert downstream["inputs"] == {"from_seed": "the seed value"}


@pytest.mark.asyncio
async def test_params_are_absent_by_default() -> None:
    assert a_context().params == {}
