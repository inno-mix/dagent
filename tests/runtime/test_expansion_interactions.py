"""Expansion against the rest of the engine — written from FR-7's text, not from the code.

The Phase 6 tests check that expansion works. These check the clause that is easiest to
satisfy on paper and hardest to satisfy in practice: *"Expansion must not deadlock or
starve."* A graph that grows while it is running has to keep working with everything that
was already true of it — concurrency caps, failure modes, budgets, timeouts, resume — and
each of those is a place where a new node can be quietly stranded.

Every test here has a wall-clock timeout, because the failure mode under investigation is
a hang, and a hanging test that never fails is worth nothing.
"""

import asyncio

import pytest

from dagent.agents.fake import FailingAgent, FakeAgent, HangingAgent
from dagent.errors import AgentError
from dagent.graph.builder import WorkflowBuilder, build_node
from dagent.models.state import NodeOutput, NodeState, RunState
from dagent.models.workflow import Policy, Workflow
from dagent.policy.limits import Budget, Limits
from dagent.policy.retry import Backoff, no_jitter
from dagent.policy.run import FailureMode, RunPolicy
from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import ManualClock
from dagent.runtime.executor import Executor
from dagent.runtime.model import StubModelClient
from dagent.runtime.registry import AgentRegistry
from dagent.store.memory import InMemoryStateStore

TIMEOUT = 5


def registry_with(**agents: object) -> AgentRegistry:
    registry = AgentRegistry()
    for name, factory in agents.items():
        registry.add(name, factory)  # type: ignore[arg-type]
    return registry


class Adds:
    """Expands with whatever nodes it is handed."""

    def __init__(self, *nodes: object) -> None:
        self._nodes = nodes

    async def run(self, ctx: AgentContext) -> NodeOutput:
        ctx.expand(*self._nodes)  # type: ignore[arg-type]
        return ctx.node_id


async def drive(workflow: Workflow, registry: AgentRegistry, **kwargs: object) -> object:
    return await asyncio.wait_for(
        Executor(store=InMemoryStateStore(), registry=registry, **kwargs).run(  # type: ignore[arg-type]
            workflow, run_id="r1"
        ),
        timeout=TIMEOUT,
    )


# --- true concurrency -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_planners_expanding_at_the_same_instant_both_land() -> None:
    # `RunGraph.apply` is only atomic because it never suspends. This forces both planners
    # to be *inside* their run methods simultaneously, so if the merge were interleavable
    # one expansion would overwrite the other and a node would go missing.
    barrier = asyncio.Barrier(2)

    class Simultaneous:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            await barrier.wait()
            ctx.expand(build_node(f"{ctx.node_id}.child", "fake", depends_on=[ctx.node_id]))
            return None

    workflow = (
        WorkflowBuilder("two-planners").add_node("p1", "planner").add_node("p2", "planner").build()
    )

    run = await drive(workflow, registry_with(planner=Simultaneous, fake=FakeAgent))

    assert sorted(run.nodes) == ["p1", "p1.child", "p2", "p2.child"]  # type: ignore[attr-defined]
    assert run.state is RunState.SUCCEEDED  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_new_node_may_wait_on_a_node_that_is_still_running() -> None:
    # The obvious deadlock: the added node's dependency has not finished, so it cannot be
    # ready yet. It has to become ready later rather than never.
    released = asyncio.Event()

    class Slow:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            await released.wait()
            return "slow"

    class ExpandsOntoSlow:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            ctx.expand(build_node("after_slow", "fake", inputs={"x": "slow"}))
            released.set()
            return None

    workflow = (
        WorkflowBuilder("onto-running").add_node("slow", "slow").add_node("plan", "planner").build()
    )

    run = await drive(workflow, registry_with(planner=ExpandsOntoSlow, slow=Slow, fake=FakeAgent))

    assert run.nodes["after_slow"].state is NodeState.SUCCESS  # type: ignore[attr-defined]
    assert run.state is RunState.SUCCEEDED  # type: ignore[attr-defined]


# --- starvation -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_added_nodes_are_not_starved_by_a_concurrency_cap_of_one() -> None:
    # With a single permit, the planner holds the only slot while it expands. If the
    # permit were held across the merge, or the loop stopped offering new nodes once it
    # had seen an empty ready set, the added nodes would never run.
    workers = [build_node(f"w{index}", "fake", depends_on=["plan"]) for index in range(4)]

    run = await drive(
        WorkflowBuilder("capped").add_node("plan", "planner").build(),
        registry_with(planner=lambda: Adds(*workers), fake=FakeAgent),
        policy=RunPolicy(limits=Limits(max_concurrency=1)),
    )

    assert run.state is RunState.SUCCEEDED  # type: ignore[attr-defined]
    assert len(run.nodes) == 5  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_chain_of_added_nodes_runs_in_dependency_order() -> None:
    order: list[str] = []

    class Recording:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            order.append(ctx.node_id)
            return ctx.node_id

    chain = [
        build_node("s1", "fake", depends_on=["plan"]),
        build_node("s2", "fake", inputs={"x": "s1"}),
        build_node("s3", "fake", inputs={"x": "s2"}),
    ]

    run = await drive(
        WorkflowBuilder("chained").add_node("plan", "planner").build(),
        registry_with(planner=lambda: Adds(*chain), fake=Recording),
    )

    assert order == ["s1", "s2", "s3"]
    assert run.state is RunState.SUCCEEDED  # type: ignore[attr-defined]


# --- failure semantics over a grown graph ------------------------------------------------


@pytest.mark.asyncio
async def test_skip_downstream_reaches_nodes_that_did_not_exist_at_submit_time() -> None:
    # `descendants` is computed over the live graph, so the blast radius of a failure has
    # to include nodes an expansion added after the run started.
    added = [
        build_node("bad", "boom", depends_on=["plan"]),
        build_node("downstream", "fake", inputs={"x": "bad"}),
    ]

    run = await drive(
        WorkflowBuilder("skip").add_node("plan", "planner").build(),
        registry_with(planner=lambda: Adds(*added), boom=FailingAgent, fake=FakeAgent),
        policy=RunPolicy(failure_mode=FailureMode.SKIP_DOWNSTREAM),
    )

    assert run.nodes["bad"].state is NodeState.FAILED  # type: ignore[attr-defined]
    assert run.nodes["downstream"].state is NodeState.SKIPPED  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_fail_fast_stops_a_run_that_has_already_grown() -> None:
    added = [
        build_node("bad", "boom", depends_on=["plan"]),
        build_node("slow", "hang", depends_on=["plan"]),
    ]

    run = await drive(
        WorkflowBuilder("ff").add_node("plan", "planner").build(),
        registry_with(planner=lambda: Adds(*added), boom=FailingAgent, hang=HangingAgent),
        policy=RunPolicy(failure_mode=FailureMode.FAIL_FAST),
    )

    assert run.state is RunState.FAILED  # type: ignore[attr-defined]
    assert run.nodes["slow"].state is NodeState.FAILED  # type: ignore[attr-defined]
    assert "fail_fast" in (run.nodes["slow"].error or "")  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_timeout_on_an_added_node_behaves_like_any_other() -> None:
    added = [build_node("slow", "hang", depends_on=["plan"], policy=Policy(timeout_s=0.05))]

    run = await drive(
        WorkflowBuilder("to").add_node("plan", "planner").build(),
        registry_with(planner=lambda: Adds(*added), hang=HangingAgent),
    )

    assert run.nodes["slow"].state is NodeState.FAILED  # type: ignore[attr-defined]
    assert "TimeoutError" in (run.nodes["slow"].error or "")  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_the_budget_still_binds_over_nodes_that_were_added_later() -> None:
    class RealCaller:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            from dagent.models.model_call import ModelRequest

            response = await ctx.model.complete(ModelRequest(prompt=f"work {ctx.node_id}"))
            return response.text

    added = [build_node(f"c{index}", "caller", depends_on=["plan"]) for index in range(3)]
    budget = Budget(max_tokens=4)

    run = await drive(
        WorkflowBuilder("budgeted").add_node("plan", "planner").build(),
        registry_with(planner=lambda: Adds(*added), caller=RealCaller),
        model=StubModelClient(lambda request: "one two three"),
        policy=RunPolicy(budget=budget),
    )

    assert run.state is RunState.BUDGET_EXCEEDED  # type: ignore[attr-defined]
    assert budget.refused


# --- resume over a grown graph ------------------------------------------------------------


class Crash(BaseException):
    """Stands in for the process dying."""


class CrashAt(InMemoryStateStore):
    def __init__(self, node_id: str, state: NodeState) -> None:
        super().__init__()
        self._target = (node_id, state)
        self.armed = True

    async def save_node_state(self, record):  # type: ignore[no-untyped-def]
        if self.armed and (record.node_id, record.state) == self._target:
            self.armed = False
            raise Crash("process died")
        await super().save_node_state(record)


@pytest.mark.asyncio
async def test_a_crash_between_expanding_and_succeeding_replays_the_expansion() -> None:
    # The nastiest ordering: the graph was persisted, then the process died before the
    # planner was marked SUCCESS. Resume re-runs the planner, which restates the same
    # nodes — and that must be a no-op, not a second fan-out.
    added = [build_node("child", "fake", depends_on=["plan"])]
    store = CrashAt("plan", NodeState.SUCCESS)
    registry = registry_with(planner=lambda: Adds(*added), fake=FakeAgent)
    workflow = WorkflowBuilder("crashy").add_node("plan", "planner").build()

    with pytest.raises(Crash):
        await Executor(registry=registry, store=store).run(workflow, run_id="r1")

    mid = await store.load_workflow("r1")
    assert [node.id for node in mid.nodes] == ["plan", "child"]

    resumed = await asyncio.wait_for(
        Executor(registry=registry, store=store).resume("r1"), timeout=TIMEOUT
    )

    assert resumed.state is RunState.SUCCEEDED
    assert [node.id for node in (await store.load_workflow("r1")).nodes] == ["plan", "child"]


@pytest.mark.asyncio
async def test_a_planner_that_replans_differently_after_a_crash_does_not_deadlock() -> None:
    # The documented hazard: a model-backed planner re-prompted after a crash comes back
    # with a different plan, so the replay is not a no-op. Whatever the engine does with
    # the leftovers, it must still terminate and it must still be honest about it.
    class Replanning:
        calls = 0

        async def run(self, ctx: AgentContext) -> NodeOutput:
            Replanning.calls += 1
            name = "first" if Replanning.calls == 1 else "second"
            ctx.expand(build_node(name, "fake", depends_on=[ctx.node_id]))
            return None

    Replanning.calls = 0
    store = CrashAt("plan", NodeState.SUCCESS)
    registry = registry_with(planner=Replanning, fake=FakeAgent)
    workflow = WorkflowBuilder("replan").add_node("plan", "planner").build()

    with pytest.raises(Crash):
        await Executor(registry=registry, store=store).run(workflow, run_id="r1")

    resumed = await asyncio.wait_for(
        Executor(registry=registry, store=store).resume("r1"), timeout=TIMEOUT
    )

    # Both plans are in the graph — the first was already durable when the process died.
    # This pins the behaviour rather than endorsing it; see the note in the report.
    assert sorted(resumed.nodes) == ["first", "plan", "second"]
    assert resumed.state is RunState.SUCCEEDED


# --- odds and ends -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_planner_that_expands_with_nothing_is_an_ordinary_node() -> None:
    run = await drive(
        WorkflowBuilder("noop").add_node("plan", "planner").build(),
        registry_with(planner=lambda: Adds()),
    )

    assert run.state is RunState.SUCCEEDED  # type: ignore[attr-defined]
    assert list(run.nodes) == ["plan"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_two_planners_asking_for_the_same_new_node_id_conflict_loudly() -> None:
    # Not a hypothetical: two planners deriving ids from a shared subtopic would collide.
    # The second must fail rather than silently overwrite the first's definition.
    workflow = WorkflowBuilder("collide").add_node("p1", "one").add_node("p2", "two").build()
    registry = registry_with(
        one=lambda: Adds(build_node("shared", "fake", params={"from": "p1"})),
        two=lambda: Adds(build_node("shared", "fake", params={"from": "p2"})),
        fake=FakeAgent,
    )

    run = await drive(workflow, registry)

    outcomes = {run.nodes["p1"].state, run.nodes["p2"].state}  # type: ignore[attr-defined]
    assert outcomes == {NodeState.SUCCESS, NodeState.FAILED}
    loser = next(r for r in run.nodes.values() if r.state is NodeState.FAILED)  # type: ignore[attr-defined]
    assert "redefines node 'shared'" in (loser.error or "")
    assert run.state is RunState.FAILED  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_retried_planner_that_replans_within_one_run_replaces_nothing() -> None:
    # Same shape as the crash case but without the crash: attempt 0 fails after asking,
    # attempt 1 asks for something else. Only the winning plan should exist.
    class Flaky:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            ctx.expand(build_node(f"take_{ctx.attempt}", "fake", depends_on=[ctx.node_id]))
            if ctx.attempt == 0:
                raise AgentError("try again")
            return None

    run = await drive(
        WorkflowBuilder("flaky")
        .add_node("plan", "planner", policy=Policy(max_attempts=2, backoff_initial_s=1.0))
        .build(),
        registry_with(planner=Flaky, fake=FakeAgent),
        clock=ManualClock(),
        policy=RunPolicy(backoff=Backoff(jitter=no_jitter)),
    )

    assert sorted(run.nodes) == ["plan", "take_1"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_an_expansion_cannot_smuggle_in_an_edge_onto_an_existing_node() -> None:
    # The strand guardrail, stated as an attack rather than as a property: restate an
    # existing node with an extra dependency and see whether it takes.
    workflow = (
        WorkflowBuilder("smuggle").add_node("plan", "planner").add_node("victim", "fake").build()
    )
    attack = build_node("victim", "fake", depends_on=["plan"])

    run = await drive(workflow, registry_with(planner=lambda: Adds(attack), fake=FakeAgent))

    assert run.nodes["plan"].state is NodeState.FAILED  # type: ignore[attr-defined]
    assert "redefines node 'victim'" in (run.nodes["plan"].error or "")  # type: ignore[attr-defined]
    assert run.nodes["victim"].state is NodeState.SUCCESS  # type: ignore[attr-defined]
