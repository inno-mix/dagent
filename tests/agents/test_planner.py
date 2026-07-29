"""The planner: what it asks the model, what it parses, and what it emits.

Network-free like everything else — the agent under test is the real agent, only the
transport is a `StubModelClient`.
"""

import pytest

from dagent.agents.planner import MAX_SUBTOPICS, PlannerAgent
from dagent.errors import AgentError
from dagent.models.state import NodeOutput
from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import ManualClock
from dagent.runtime.model import StubModelClient

THREE = "attention mechanisms\ninstruction tuning\nreinforcement learning from feedback"


def a_context(
    reply: str = THREE,
    *,
    node_id: str = "plan",
    question: str = "how do LLMs work?",
    **params: object,
) -> AgentContext:
    return AgentContext(
        run_id="r1",
        node_id=node_id,
        attempt=0,
        inputs={"question": question},
        params=dict(params),  # type: ignore[arg-type]
        clock=ManualClock(),
        model=StubModelClient((reply,)),
    )


async def plan(ctx: AgentContext) -> NodeOutput:
    return await PlannerAgent().run(ctx)


@pytest.mark.asyncio
async def test_the_planner_reports_the_subtopics_it_chose() -> None:
    ctx = a_context(subtopics=3)

    output = await plan(ctx)

    assert isinstance(output, dict)
    assert output["subtopics"] == [
        "attention mechanisms",
        "instruction tuning",
        "reinforcement learning from feedback",
    ]


@pytest.mark.asyncio
async def test_the_planner_emits_one_researcher_per_subtopic_plus_a_synthesizer() -> None:
    ctx = a_context(subtopics=3)

    await plan(ctx)

    assert [node.id for node in ctx.expansion.nodes] == [
        "plan.research_0",
        "plan.research_1",
        "plan.research_2",
        "plan.synthesis",
    ]
    assert [node.agent for node in ctx.expansion.nodes] == [
        "researcher",
        "researcher",
        "researcher",
        "synthesizer",
    ]


@pytest.mark.asyncio
async def test_each_researcher_carries_its_own_subtopic_as_a_parameter() -> None:
    # Not as an input: there is no upstream node holding the topic, and `params` is what
    # the definition carries. ARCHITECTURE §2 said this is what params were for.
    ctx = a_context(subtopics=3)

    await plan(ctx)

    assert [node.params["topic"] for node in ctx.expansion.nodes[:3]] == [
        "attention mechanisms",
        "instruction tuning",
        "reinforcement learning from feedback",
    ]


@pytest.mark.asyncio
async def test_the_researchers_wait_for_the_planner() -> None:
    # Ordering only — they read nothing from it — but a node that ran before its planner
    # would be a node the planner had not yet decided to create.
    ctx = a_context(subtopics=2)

    await plan(ctx)

    assert all(node.depends_on == ("plan",) for node in ctx.expansion.nodes[:2])


@pytest.mark.asyncio
async def test_the_synthesizer_fans_in_every_researcher() -> None:
    ctx = a_context(subtopics=3)

    await plan(ctx)

    synthesis = ctx.expansion.nodes[-1]
    assert set(synthesis.inputs.values()) == {
        "plan.research_0",
        "plan.research_1",
        "plan.research_2",
    }
    assert set(synthesis.depends_on) == set(synthesis.inputs.values())


@pytest.mark.asyncio
async def test_the_generated_ids_derive_from_the_planners_own_id() -> None:
    # Which is what makes a re-executed planner restate the same nodes, so its replayed
    # expansion is a no-op rather than a second fan-out (DR-4).
    first = a_context(node_id="alpha", subtopics=2)
    second = a_context(node_id="alpha", subtopics=2)

    await plan(first)
    await plan(second)

    assert [node.id for node in first.expansion.nodes] == [
        node.id for node in second.expansion.nodes
    ]
    assert first.expansion.nodes == second.expansion.nodes


@pytest.mark.asyncio
async def test_the_planner_asks_for_the_number_of_subtopics_it_was_configured_with() -> None:
    ctx = a_context(subtopics=2)

    await plan(ctx)

    assert "Give 2 subtopics" in ctx.model.requests[0].prompt  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_extra_lines_beyond_what_was_asked_for_are_dropped() -> None:
    # The bound has to hold against the model, not merely against the prompt.
    ctx = a_context("one\ntwo\nthree\nfour\nfive", subtopics=2)

    await plan(ctx)

    assert len(ctx.expansion.nodes) == 3  # two researchers and the synthesizer


@pytest.mark.asyncio
async def test_numbering_and_bullets_are_stripped() -> None:
    ctx = a_context("1. first thing\n- second thing\n* third thing", subtopics=3)

    output = await plan(ctx)

    assert isinstance(output, dict)
    assert output["subtopics"] == ["first thing", "second thing", "third thing"]


@pytest.mark.asyncio
async def test_blank_lines_are_ignored() -> None:
    ctx = a_context("alpha\n\n\nbeta\n", subtopics=3)

    output = await plan(ctx)

    assert isinstance(output, dict)
    assert output["subtopics"] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_a_repeated_subtopic_is_not_researched_twice() -> None:
    # Duplicate ids would be rejected by the expansion anyway; better to spend one call
    # than to fail the whole node over a model repeating itself.
    ctx = a_context("same\nsame\ndifferent", subtopics=3)

    output = await plan(ctx)

    assert isinstance(output, dict)
    assert output["subtopics"] == ["same", "different"]


@pytest.mark.asyncio
async def test_a_model_that_says_nothing_useful_fails_retryably() -> None:
    ctx = a_context("   \n\n  ")

    with pytest.raises(AgentError, match="no usable subtopics") as caught:
        await plan(ctx)

    assert caught.value.retryable is True


@pytest.mark.asyncio
async def test_a_planner_with_no_question_says_so() -> None:
    ctx = AgentContext(
        run_id="r1",
        node_id="plan",
        attempt=0,
        inputs={},
        clock=ManualClock(),
        model=StubModelClient((THREE,)),
    )

    with pytest.raises(AgentError, match="exactly one input") as caught:
        await plan(ctx)

    assert caught.value.retryable is False


@pytest.mark.parametrize("wanted", [0, -1, MAX_SUBTOPICS + 1, "three", 2.5, True])
@pytest.mark.asyncio
async def test_a_nonsensical_subtopic_count_is_refused(wanted: object) -> None:
    # Including `True`, which is an int as far as Python is concerned and a bug as far as
    # anyone else is.
    ctx = a_context(subtopics=wanted)

    with pytest.raises(AgentError, match="subtopics") as caught:
        await plan(ctx)

    assert caught.value.retryable is False


@pytest.mark.asyncio
async def test_the_default_subtopic_count_applies_when_none_is_given() -> None:
    ctx = a_context()

    await plan(ctx)

    assert len(ctx.expansion.nodes) == 4  # three researchers and the synthesizer


@pytest.mark.asyncio
async def test_the_planner_records_which_provider_answered() -> None:
    ctx = a_context()

    output = await plan(ctx)

    assert isinstance(output, dict)
    assert output["provider"] == "stub"
