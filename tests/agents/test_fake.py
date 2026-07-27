import pytest

from dagent.agents.fake import EchoAgent, FailingAgent, FakeAgent
from dagent.runtime.agent import Agent, AgentContext
from dagent.runtime.clock import ManualClock


def a_context(node_id: str = "a", **inputs: object) -> AgentContext:
    return AgentContext(
        run_id="r1",
        node_id=node_id,
        attempt=0,
        inputs=dict(inputs),  # type: ignore[arg-type]
        clock=ManualClock(),
    )


@pytest.mark.parametrize("agent_type", [FakeAgent, EchoAgent, FailingAgent])
def test_every_fake_satisfies_the_agent_protocol(agent_type: type) -> None:
    agent: Agent = agent_type()
    assert agent is not None


@pytest.mark.asyncio
async def test_fake_agent_reports_its_node_and_inputs() -> None:
    output = await FakeAgent().run(a_context("b", upstream={"x": 1}))

    assert output == {"node_id": "b", "attempt": 0, "inputs": {"upstream": {"x": 1}}}


@pytest.mark.asyncio
async def test_fake_agent_is_deterministic() -> None:
    # Phase 5 compares an interrupted run against an uninterrupted one byte for byte.
    first = await FakeAgent().run(a_context("b", upstream="v"))
    second = await FakeAgent().run(a_context("b", upstream="v"))

    assert first == second


@pytest.mark.asyncio
async def test_fake_agent_orders_inputs_so_output_does_not_depend_on_insertion_order() -> None:
    forwards = await FakeAgent().run(a_context("b", alpha=1, beta=2))
    backwards = await FakeAgent().run(a_context("b", beta=2, alpha=1))

    assert forwards == backwards


@pytest.mark.asyncio
async def test_echo_agent_passes_its_single_input_through() -> None:
    assert await EchoAgent().run(a_context("b", only={"deep": [1, 2]})) == {"deep": [1, 2]}


@pytest.mark.asyncio
async def test_echo_agent_emits_none_when_it_is_a_source() -> None:
    assert await EchoAgent().run(a_context("a")) is None


@pytest.mark.asyncio
async def test_failing_agent_raises_and_names_the_node() -> None:
    with pytest.raises(RuntimeError, match="'b'"):
        await FailingAgent().run(a_context("b"))
