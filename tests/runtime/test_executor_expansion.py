"""Phase 6 acceptance: a graph that grows while it runs, and one that is refused.

The two halves of the criterion. A planner fans out to N nodes with N decided at run time
and a synthesizer fans them back in; and a planner that would introduce a cycle is
rejected without deadlocking the run — it fails, and everything else finishes.
"""

import asyncio

import pytest

from dagent.agents.fake import FailingAgent, FakeAgent
from dagent.errors import ValidationError
from dagent.graph.builder import WorkflowBuilder, build_node
from dagent.models.state import NodeOutput, NodeState, RunState
from dagent.policy.run import RunPolicy
from dagent.runtime.agent import AgentContext
from dagent.runtime.executor import Executor
from dagent.runtime.registry import AgentRegistry
from dagent.store.memory import InMemoryStateStore


def registry_with(**agents: object) -> AgentRegistry:
    registry = AgentRegistry()
    for name, factory in agents.items():
        registry.add(name, factory)  # type: ignore[arg-type]
    return registry


class Fanning:
    """Emits `width` workers and one node that fans them back in."""

    def __init__(self, width: int) -> None:
        self._width = width

    async def run(self, ctx: AgentContext) -> NodeOutput:
        workers = [
            build_node(
                f"{ctx.node_id}.worker_{index}",
                "fake",
                depends_on=[ctx.node_id],
                params={"index": index},
            )
            for index in range(self._width)
        ]
        ctx.expand(
            *workers,
            build_node(
                f"{ctx.node_id}.join",
                "fake",
                inputs={node.id: node.id for node in workers},
            ),
        )
        return {"width": self._width}


def one_planner(name: str = "plan") -> object:
    return WorkflowBuilder("dynamic").add_node(name, "planner").build()


# --- the graph grows -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_planner_fans_out_and_a_synthesizer_fans_back_in() -> None:
    store = InMemoryStateStore()
    executor = Executor(
        registry=registry_with(planner=lambda: Fanning(4), fake=FakeAgent), store=store
    )

    run = await asyncio.wait_for(executor.run(one_planner(), run_id="r1"), timeout=5)  # type: ignore[arg-type]

    assert run.state is RunState.SUCCEEDED
    assert sorted(run.nodes) == [
        "plan",
        "plan.join",
        "plan.worker_0",
        "plan.worker_1",
        "plan.worker_2",
        "plan.worker_3",
    ]
    # The fan-in really did receive every branch.
    join = await store.load_output("r1", "plan.join")
    assert isinstance(join, dict)
    assert len(join["inputs"]) == 4  # type: ignore[arg-type]


@pytest.mark.parametrize("width", [1, 2, 7])
@pytest.mark.asyncio
async def test_the_width_is_decided_at_run_time(width: int) -> None:
    # Nothing about the workflow changes between these runs. The number of nodes is a
    # property of what the planner said, not of what the author wrote down.
    store = InMemoryStateStore()
    executor = Executor(
        registry=registry_with(planner=lambda: Fanning(width), fake=FakeAgent), store=store
    )

    run = await asyncio.wait_for(executor.run(one_planner(), run_id="r1"), timeout=5)  # type: ignore[arg-type]

    assert run.state is RunState.SUCCEEDED
    assert len(run.nodes) == width + 2  # planner, N workers, the join


@pytest.mark.asyncio
async def test_the_added_nodes_run_concurrently() -> None:
    # Expansion does not cost the fan-out its parallelism: the new nodes go through the
    # same ready set as any other, so they are dispatched together.
    probe = {"in_flight": 0, "peak": 0}

    class Overlapping:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            probe["in_flight"] += 1
            probe["peak"] = max(probe["peak"], probe["in_flight"])
            for _ in range(3):
                await asyncio.sleep(0)
            probe["in_flight"] -= 1
            return ctx.node_id

    executor = Executor(
        registry=registry_with(planner=lambda: Fanning(5), fake=Overlapping),
        store=InMemoryStateStore(),
    )

    await asyncio.wait_for(executor.run(one_planner(), run_id="r1"), timeout=5)  # type: ignore[arg-type]

    assert probe["peak"] == 5


@pytest.mark.asyncio
async def test_the_augmented_graph_is_persisted() -> None:
    # Or a resumed run would rebuild the graph as it was submitted and lose the plan.
    store = InMemoryStateStore()
    executor = Executor(
        registry=registry_with(planner=lambda: Fanning(2), fake=FakeAgent), store=store
    )

    await executor.run(one_planner(), run_id="r1")  # type: ignore[arg-type]

    stored = await store.load_workflow("r1")
    assert [node.id for node in stored.nodes] == [
        "plan",
        "plan.worker_0",
        "plan.worker_1",
        "plan.join",
    ]


@pytest.mark.asyncio
async def test_added_nodes_appear_in_the_run_record_before_they_run() -> None:
    class Watching(InMemoryStateStore):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[tuple[str, NodeState]] = []

        async def save_node_state(self, record):  # type: ignore[no-untyped-def]
            self.seen.append((record.node_id, record.state))
            await super().save_node_state(record)

    store = Watching()
    executor = Executor(
        registry=registry_with(planner=lambda: Fanning(1), fake=FakeAgent), store=store
    )

    await executor.run(one_planner(), run_id="r1")  # type: ignore[arg-type]

    worker = [state for node_id, state in store.seen if node_id == "plan.worker_0"]
    assert worker[0] is NodeState.PENDING


# --- the graph is refused --------------------------------------------------------------


class Cyclic:
    """Asks for two nodes that depend on each other."""

    async def run(self, ctx: AgentContext) -> NodeOutput:
        ctx.expand(
            build_node("loop_a", "fake", depends_on=["loop_b"]),
            build_node("loop_b", "fake", depends_on=["loop_a"]),
        )
        return None


@pytest.mark.asyncio
async def test_a_planner_that_would_introduce_a_cycle_fails_without_deadlocking() -> None:
    # The whole second half of the acceptance criterion. The rejection has to be a node
    # outcome, not a hang and not a crash — the run finishes, and says what went wrong.
    store = InMemoryStateStore()
    workflow = (
        WorkflowBuilder("bad-plan")
        .add_node("plan", "planner")
        .add_node("bystander", "fake")
        .build()
    )

    run = await asyncio.wait_for(
        Executor(registry=registry_with(planner=Cyclic, fake=FakeAgent), store=store).run(
            workflow, run_id="r1"
        ),
        timeout=5,
    )

    assert run.state is RunState.FAILED
    assert run.nodes["plan"].state is NodeState.FAILED
    assert "cycle" in (run.nodes["plan"].error or "")
    # Nothing else was harmed, and the run terminated on its own.
    assert run.nodes["bystander"].state is NodeState.SUCCESS


@pytest.mark.asyncio
async def test_a_rejected_expansion_adds_nothing_to_the_graph() -> None:
    store = InMemoryStateStore()

    await asyncio.wait_for(
        Executor(registry=registry_with(planner=Cyclic, fake=FakeAgent), store=store).run(
            WorkflowBuilder("bad-plan").add_node("plan", "planner").build(), run_id="r1"
        ),
        timeout=5,
    )

    assert [node.id for node in (await store.load_workflow("r1")).nodes] == ["plan"]


@pytest.mark.asyncio
async def test_a_rejected_expansion_is_not_retried() -> None:
    # A ValidationError is not weather. Retrying a planner that emits a cycle produces the
    # same cycle three times over.
    calls: list[int] = []

    class CountedCyclic(Cyclic):
        async def run(self, ctx: AgentContext) -> NodeOutput:
            calls.append(ctx.attempt)
            return await super().run(ctx)

    from dagent.models.workflow import Policy

    workflow = (
        WorkflowBuilder("bad-plan")
        .add_node("plan", "planner", policy=Policy(max_attempts=4, backoff_initial_s=1.0))
        .build()
    )

    await Executor(
        registry=registry_with(planner=CountedCyclic),
        store=InMemoryStateStore(),
    ).run(workflow, run_id="r1")

    assert calls == [0]


@pytest.mark.asyncio
async def test_a_planner_that_names_an_unregistered_agent_fails_cleanly() -> None:
    class Bogus:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            ctx.expand(build_node("child", "no_such_agent"))
            return None

    run = await asyncio.wait_for(
        Executor(registry=registry_with(planner=Bogus), store=InMemoryStateStore()).run(
            WorkflowBuilder("bad").add_node("plan", "planner").build(), run_id="r1"
        ),
        timeout=5,
    )

    assert run.nodes["plan"].state is NodeState.FAILED
    assert "not registered" in (run.nodes["plan"].error or "")


@pytest.mark.asyncio
async def test_expansion_past_the_depth_limit_is_refused() -> None:
    class Recursive:
        """Emits a copy of itself — the runaway the depth bound exists for."""

        async def run(self, ctx: AgentContext) -> NodeOutput:
            ctx.expand(build_node(f"{ctx.node_id}.child", "planner", depends_on=[ctx.node_id]))
            return None

    store = InMemoryStateStore()
    run = await asyncio.wait_for(
        Executor(registry=registry_with(planner=Recursive), store=store).run(
            WorkflowBuilder("recursive").add_node("plan", "planner").build(), run_id="r1"
        ),
        timeout=5,
    )

    # One generation allowed, the next refused: the run stops growing and terminates.
    assert run.nodes["plan"].state is NodeState.SUCCESS
    assert run.nodes["plan.child"].state is NodeState.FAILED
    assert "depth" in (run.nodes["plan.child"].error or "")
    assert run.state is RunState.FAILED


@pytest.mark.asyncio
async def test_raising_the_depth_limit_lets_a_second_generation_through() -> None:
    class Once:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            if ctx.node_id.count(".") < 1:
                ctx.expand(build_node(f"{ctx.node_id}.child", "planner", depends_on=[ctx.node_id]))
            return None

    run = await asyncio.wait_for(
        Executor(
            registry=registry_with(planner=Once),
            store=InMemoryStateStore(),
            policy=RunPolicy(max_expansion_depth=2),
        ).run(WorkflowBuilder("deep").add_node("plan", "planner").build(), run_id="r1"),
        timeout=5,
    )

    assert run.state is RunState.SUCCEEDED
    assert sorted(run.nodes) == ["plan", "plan.child"]


@pytest.mark.asyncio
async def test_expansion_is_off_when_the_depth_limit_is_zero() -> None:
    run = await asyncio.wait_for(
        Executor(
            registry=registry_with(planner=lambda: Fanning(2), fake=FakeAgent),
            store=InMemoryStateStore(),
            policy=RunPolicy(max_expansion_depth=0),
        ).run(one_planner(), run_id="r1"),  # type: ignore[arg-type]
        timeout=5,
    )

    assert run.nodes["plan"].state is NodeState.FAILED
    assert len(run.nodes) == 1


@pytest.mark.asyncio
async def test_an_attempt_that_expands_and_then_fails_leaves_nothing_behind() -> None:
    # The request is collected on the context and applied only on success, so the retry
    # starts from a graph nobody has already grown.
    #
    # The two attempts ask for *different* nodes on purpose. A planner backed by a model
    # is exactly that: re-prompted, it comes back with a different plan. Had both attempts
    # asked for the same node, the "identical restatement is a no-op" rule would mask the
    # bug — an earlier version of this test did precisely that and passed against an
    # implementation that expanded on failure too.
    from dagent.errors import AgentError
    from dagent.models.workflow import Policy
    from dagent.policy.retry import Backoff, no_jitter
    from dagent.runtime.clock import ManualClock

    class ExpandsThenFails:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            ctx.expand(
                build_node(f"{ctx.node_id}.take_{ctx.attempt}", "fake", depends_on=[ctx.node_id])
            )
            if ctx.attempt == 0:
                raise AgentError("not this time")
            return None

    store = InMemoryStateStore()
    workflow = (
        WorkflowBuilder("retry")
        .add_node("plan", "planner", policy=Policy(max_attempts=2, backoff_initial_s=1.0))
        .build()
    )

    run = await Executor(
        registry=registry_with(planner=ExpandsThenFails, fake=FakeAgent),
        store=store,
        clock=ManualClock(),
        policy=RunPolicy(backoff=Backoff(jitter=no_jitter)),
    ).run(workflow, run_id="r1")

    assert run.state is RunState.SUCCEEDED
    # Only the winning attempt's node. The abandoned attempt's `take_0` is nowhere.
    assert [node.id for node in (await store.load_workflow("r1")).nodes] == [
        "plan",
        "plan.take_1",
    ]


# --- expansion and the rest of the engine ----------------------------------------------


@pytest.mark.asyncio
async def test_a_crashed_run_resumes_into_the_expanded_graph() -> None:
    # The reason Phase 5 persists the definition: no file describes this graph.
    class Crash(BaseException):
        pass

    class Crashing(InMemoryStateStore):
        def __init__(self) -> None:
            super().__init__()
            self.armed = True

        async def save_node_state(self, record):  # type: ignore[no-untyped-def]
            if self.armed and record.node_id == "plan.join":
                self.armed = False
                raise Crash("process died")
            await super().save_node_state(record)

    store = Crashing()
    registry = registry_with(planner=lambda: Fanning(2), fake=FakeAgent)

    with pytest.raises(Crash):
        await Executor(registry=registry, store=store).run(one_planner(), run_id="r1")  # type: ignore[arg-type]

    resumed = await Executor(registry=registry, store=store).resume("r1")

    assert resumed.state is RunState.SUCCEEDED
    assert sorted(resumed.nodes) == ["plan", "plan.join", "plan.worker_0", "plan.worker_1"]


@pytest.mark.asyncio
async def test_a_failure_inside_an_expanded_branch_behaves_like_any_other() -> None:
    class FanningToFailure:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            ctx.expand(
                build_node("bad", "boom", depends_on=[ctx.node_id]),
                build_node("after", "fake", inputs={"x": "bad"}),
            )
            return None

    run = await asyncio.wait_for(
        Executor(
            registry=registry_with(planner=FanningToFailure, boom=FailingAgent, fake=FakeAgent),
            store=InMemoryStateStore(),
        ).run(WorkflowBuilder("dyn").add_node("plan", "planner").build(), run_id="r1"),
        timeout=5,
    )

    assert run.nodes["bad"].state is NodeState.FAILED
    assert run.nodes["after"].state is NodeState.PENDING
    assert run.state is RunState.FAILED


def test_expansion_is_reachable_from_the_agent_context() -> None:
    # FR-7 is an agent-facing feature; if it needs an import from the engine it is not
    # one. `ctx.expand` is the whole surface.
    assert callable(AgentContext.expand)


@pytest.mark.asyncio
async def test_the_node_ceiling_stops_a_planner_that_asks_for_too_much() -> None:
    run = await asyncio.wait_for(
        Executor(
            registry=registry_with(planner=lambda: Fanning(50), fake=FakeAgent),
            store=InMemoryStateStore(),
            policy=RunPolicy(max_graph_nodes=10),
        ).run(one_planner(), run_id="r1"),  # type: ignore[arg-type]
        timeout=5,
    )

    assert run.nodes["plan"].state is NodeState.FAILED
    assert "above the limit of 10" in (run.nodes["plan"].error or "")


@pytest.mark.asyncio
async def test_a_workflow_with_no_planner_is_completely_unaffected() -> None:
    # Expansion is a capability, not a tax: a static graph runs exactly as it did before.
    workflow = (
        WorkflowBuilder("static")
        .add_node("a", "fake")
        .add_node("b", "fake", inputs={"x": "a"})
        .build()
    )
    store = InMemoryStateStore()

    run = await Executor(registry=registry_with(fake=FakeAgent), store=store).run(
        workflow, run_id="r1"
    )

    assert run.state is RunState.SUCCEEDED
    assert await store.load_workflow("r1") == workflow


@pytest.mark.asyncio
async def test_an_expansion_naming_a_node_that_does_not_exist_is_refused() -> None:
    class Dangling:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            ctx.expand(build_node("child", "fake", inputs={"x": "ghost"}))
            return None

    run = await Executor(
        registry=registry_with(planner=Dangling, fake=FakeAgent), store=InMemoryStateStore()
    ).run(WorkflowBuilder("dangling").add_node("plan", "planner").build(), run_id="r1")

    assert run.nodes["plan"].state is NodeState.FAILED
    assert "ghost" in (run.nodes["plan"].error or "")


@pytest.mark.asyncio
async def test_the_rejection_is_a_validation_error_the_caller_can_recognise() -> None:
    graph_error: list[str] = []

    class Reporting:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            ctx.expand(build_node("child", "no_such_agent"))
            return None

    run = await Executor(registry=registry_with(planner=Reporting), store=InMemoryStateStore()).run(
        WorkflowBuilder("bad").add_node("plan", "planner").build(), run_id="r1"
    )

    graph_error.append(run.nodes["plan"].error or "")
    assert graph_error[0].startswith(ValidationError.__name__)
