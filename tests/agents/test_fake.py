import asyncio

import pytest

from dagent.agents.fake import (
    ConstantAgent,
    EchoAgent,
    FailingAgent,
    FakeAgent,
    FlakyAgent,
    HangingAgent,
)
from dagent.errors import AgentError
from dagent.policy.retry import default_retryable
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


@pytest.mark.parametrize(
    "agent_type", [FakeAgent, EchoAgent, FailingAgent, FlakyAgent, HangingAgent]
)
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


@pytest.mark.asyncio
async def test_failing_agent_raises_something_the_default_policy_will_not_retry() -> None:
    # A bug, not weather: retrying it buys three identical stack traces.
    with pytest.raises(RuntimeError) as caught:
        await FailingAgent().run(a_context("b"))

    assert default_retryable(caught.value) is False


def a_context_at(attempt: int) -> AgentContext:
    return AgentContext(run_id="r1", node_id="n", attempt=attempt, inputs={}, clock=ManualClock())


@pytest.mark.asyncio
async def test_flaky_agent_fails_below_its_threshold_attempt() -> None:
    with pytest.raises(AgentError, match="attempt 0"):
        await FlakyAgent(fail_until_attempt=2).run(a_context_at(0))


@pytest.mark.asyncio
async def test_flaky_agent_succeeds_once_the_attempt_reaches_its_threshold() -> None:
    assert await FlakyAgent(fail_until_attempt=2).run(a_context_at(2)) == {
        "node_id": "n",
        "attempt": 2,
    }


@pytest.mark.asyncio
async def test_flaky_agent_fails_retryably_so_the_default_policy_gives_it_another_go() -> None:
    with pytest.raises(AgentError) as caught:
        await FlakyAgent().run(a_context_at(0))

    assert caught.value.retryable is True


@pytest.mark.asyncio
async def test_flaky_agent_succeeds_immediately_when_nothing_is_asked_to_fail() -> None:
    assert await FlakyAgent(fail_until_attempt=0).run(a_context_at(0)) is not None


@pytest.mark.asyncio
async def test_hanging_agent_blocks_until_cancelled_and_says_so() -> None:
    agent = HangingAgent()
    task = asyncio.create_task(agent.run(a_context_at(0)))
    await asyncio.wait_for(agent.entered.wait(), timeout=5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (agent.started, agent.cancelled) == (1, 1)


@pytest.mark.asyncio
async def test_constant_agent_reports_a_missing_value_as_permanent() -> None:
    with pytest.raises(AgentError) as caught:
        await ConstantAgent().run(a_context_at(0))

    assert caught.value.retryable is False
