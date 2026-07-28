"""The two real agents, run against the stub client — no network, real agent code.

This is the "same workflow runs in tests with the fake client and zero network" half of
Phase 3's acceptance: the researcher and synthesizer here are exactly the classes the CLI
runs against Gemini, with only the transport swapped.
"""

import pytest

from dagent.agents.researcher import ResearcherAgent
from dagent.agents.synthesizer import SynthesizerAgent
from dagent.errors import AgentError
from dagent.graph.builder import WorkflowBuilder
from dagent.models.state import NodeState, RunState
from dagent.runtime.agent import Agent, AgentContext
from dagent.runtime.clock import ManualClock
from dagent.runtime.executor import Executor
from dagent.runtime.model import StubModelClient
from dagent.runtime.registry import default_registry
from dagent.store.memory import InMemoryStateStore


def a_context(node_id: str, model: StubModelClient, **inputs: object) -> AgentContext:
    return AgentContext(
        run_id="r1",
        node_id=node_id,
        attempt=0,
        inputs=dict(inputs),  # type: ignore[arg-type]
        clock=ManualClock(),
        model=model,
    )


# --- researcher ---------------------------------------------------------------------


def test_the_researcher_satisfies_the_agent_protocol() -> None:
    agent: Agent = ResearcherAgent()
    assert agent is not None


@pytest.mark.asyncio
async def test_the_researcher_returns_findings_for_its_topic() -> None:
    model = StubModelClient(["attention is all you need"])

    output = await ResearcherAgent().run(a_context("r", model, topic="transformers"))

    assert output == {
        "topic": "transformers",
        "findings": "attention is all you need",
        "provider": "stub",
        "model": "stub-model",
    }


@pytest.mark.asyncio
async def test_the_researcher_puts_its_topic_in_the_prompt() -> None:
    model = StubModelClient(["ok"])

    await ResearcherAgent().run(a_context("r", model, topic="quantum error correction"))

    assert "quantum error correction" in model.requests[0].prompt
    assert model.requests[0].system is not None


@pytest.mark.asyncio
async def test_the_researcher_does_not_care_what_its_input_is_called() -> None:
    # The node names the edge; the agent just needs the one value.
    model = StubModelClient(["ok", "ok"])

    first = await ResearcherAgent().run(a_context("r", model, topic="x"))
    second = await ResearcherAgent().run(a_context("r", model, anything_else="x"))

    assert first == second


@pytest.mark.asyncio
async def test_the_researcher_makes_exactly_one_model_call() -> None:
    model = StubModelClient(["ok"])

    await ResearcherAgent().run(a_context("r", model, topic="x"))

    assert len(model.requests) == 1


@pytest.mark.asyncio
async def test_the_researcher_refuses_ambiguous_input() -> None:
    model = StubModelClient(["ok"])

    with pytest.raises(AgentError, match="exactly one input"):
        await ResearcherAgent().run(a_context("r", model, a="1", b="2"))


@pytest.mark.asyncio
async def test_the_researcher_refuses_to_run_with_no_input() -> None:
    with pytest.raises(AgentError, match="exactly one input"):
        await ResearcherAgent().run(a_context("r", StubModelClient(["ok"])))


# --- synthesizer --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_synthesizer_merges_every_input() -> None:
    model = StubModelClient(["merged"])

    output = await SynthesizerAgent().run(a_context("s", model, a="note a", b="note b"))

    assert output == {
        "summary": "merged",
        "sources": ["a", "b"],
        "provider": "stub",
        "model": "stub-model",
    }
    assert "note a" in model.requests[0].prompt
    assert "note b" in model.requests[0].prompt


@pytest.mark.asyncio
async def test_the_synthesizer_unwraps_researcher_output_into_prose() -> None:
    model = StubModelClient(["merged"])
    researcher_output = {"topic": "t", "findings": "the findings text", "provider": "stub"}

    await SynthesizerAgent().run(a_context("s", model, a=researcher_output))

    prompt = model.requests[0].prompt
    assert "the findings text" in prompt
    assert "findings" not in prompt.replace("the findings text", "")


@pytest.mark.asyncio
async def test_the_synthesizer_prompt_is_stable_regardless_of_input_order() -> None:
    # Determinism is the prerequisite for replay: same inputs, same prompt, same bytes.
    forwards = StubModelClient(["m"])
    backwards = StubModelClient(["m"])

    await SynthesizerAgent().run(a_context("s", forwards, alpha="1", beta="2"))
    await SynthesizerAgent().run(a_context("s", backwards, beta="2", alpha="1"))

    assert forwards.requests[0].prompt == backwards.requests[0].prompt


@pytest.mark.asyncio
async def test_the_synthesizer_scales_from_two_inputs_to_many() -> None:
    # Phase 6's planner decides N at runtime; the agent must not care what N is.
    model = StubModelClient(["merged"])
    inputs = {f"branch_{index}": f"note {index}" for index in range(7)}

    output = await SynthesizerAgent().run(a_context("s", model, **inputs))

    assert isinstance(output, dict)
    assert output["sources"] == sorted(inputs)


@pytest.mark.asyncio
async def test_the_synthesizer_serializes_an_unrecognised_output_deterministically() -> None:
    # Not a string and not a researcher dict, so it is rendered as sorted JSON.
    model = StubModelClient(["m"])

    await SynthesizerAgent().run(a_context("s", model, a={"b": 2, "a": 1}))

    assert '{"a": 1, "b": 2}' in model.requests[0].prompt


@pytest.mark.asyncio
async def test_the_synthesizer_refuses_to_run_with_nothing_to_merge() -> None:
    with pytest.raises(AgentError, match="no inputs"):
        await SynthesizerAgent().run(a_context("s", StubModelClient(["ok"])))


# --- both, end to end, through the real executor ------------------------------------


@pytest.mark.asyncio
async def test_the_research_workflow_runs_with_zero_network() -> None:
    model = StubModelClient(
        lambda request: "SYNTHESIS" if "end of notes" in request.prompt else "FINDINGS"
    )
    store = InMemoryStateStore()
    workflow = (
        WorkflowBuilder("research")
        .add_node("seed_a", "constant", params={"value": "topic a"})
        .add_node("seed_b", "constant", params={"value": "topic b"})
        .add_node("research_a", "researcher", inputs={"topic": "seed_a"})
        .add_node("research_b", "researcher", inputs={"topic": "seed_b"})
        .add_node("merge", "synthesizer", inputs={"a": "research_a", "b": "research_b"})
        .build(known_agents=default_registry.names())
    )

    run = await Executor(registry=default_registry, store=store, model=model).run(
        workflow, run_id="r1"
    )

    assert run.state is RunState.SUCCEEDED
    assert {record.state for record in run.nodes.values()} == {NodeState.SUCCESS}

    merged = await store.load_output("r1", "merge")
    assert isinstance(merged, dict)
    assert merged["summary"] == "SYNTHESIS"
    assert merged["sources"] == ["a", "b"]

    # Three model calls: two researchers and one synthesizer. The constants make none.
    assert len(model.requests) == 3
