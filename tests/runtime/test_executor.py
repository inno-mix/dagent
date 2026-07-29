"""Phase 2 acceptance plus the executor's failure, cancellation, and ordering behaviour."""

import asyncio
from collections.abc import Sequence
from datetime import datetime

import pytest

from dagent.agents.fake import FailingAgent, FakeAgent
from dagent.errors import ValidationError
from dagent.graph.builder import WorkflowBuilder
from dagent.models.model_call import ModelCallRecord
from dagent.models.state import NodeOutput, NodeState, RunState, RunStateRecord
from dagent.models.workflow import Node, Workflow
from dagent.runtime.agent import Agent, AgentContext
from dagent.runtime.clock import ManualClock
from dagent.runtime.executor import Executor
from dagent.runtime.registry import AgentRegistry
from dagent.store.base import StateStore
from dagent.store.memory import InMemoryStateStore


class ConcurrencyProbe:
    """Forces overlap and records it.

    The barrier is the load-bearing part: a node cannot finish until `parties` nodes are
    inside it at once, so if the executor ran the branches one after another the run
    would never finish and the test fails on its timeout rather than passing by luck.
    """

    def __init__(self, parties: int) -> None:
        self.barrier = asyncio.Barrier(parties)
        self.intervals: dict[str, tuple[datetime, datetime]] = {}
        self.in_flight = 0
        self.peak = 0


class ProbeAgent:
    def __init__(self, probe: ConcurrencyProbe) -> None:
        self._probe = probe

    async def run(self, ctx: AgentContext) -> NodeOutput:
        started = ctx.clock.now()
        self._probe.in_flight += 1
        self._probe.peak = max(self._probe.peak, self._probe.in_flight)

        await self._probe.barrier.wait()

        self._probe.in_flight -= 1
        self._probe.intervals[ctx.node_id] = (started, ctx.clock.now())
        return {"node": ctx.node_id}


class SpyStore:
    """A StateStore that records the order of every write it is asked to make."""

    def __init__(self) -> None:
        self.inner = InMemoryStateStore()
        self.events: list[tuple[str, str, str]] = []

    async def checkpoint(self, run: RunStateRecord) -> None:
        self.events.append(("run", run.run_id, run.state))
        await self.inner.checkpoint(run)

    async def save_workflow(self, run_id: str, workflow: Workflow) -> None:
        self.events.append(("workflow", run_id, workflow.name))
        await self.inner.save_workflow(run_id, workflow)

    async def load_workflow(self, run_id: str) -> Workflow:
        return await self.inner.load_workflow(run_id)

    async def save_node_state(self, record):  # type: ignore[no-untyped-def]
        self.events.append(("state", record.node_id, record.state))
        await self.inner.save_node_state(record)

    async def append_output(self, run_id: str, node_id: str, output: NodeOutput) -> None:
        self.events.append(("output", node_id, ""))
        await self.inner.append_output(run_id, node_id, output)

    async def load_run(self, run_id: str) -> RunStateRecord:
        return await self.inner.load_run(run_id)

    async def load_output(self, run_id: str, node_id: str) -> NodeOutput:
        return await self.inner.load_output(run_id, node_id)

    async def append_model_call(self, record: ModelCallRecord) -> None:
        self.events.append(("model", record.node_id, str(record.sequence)))
        await self.inner.append_model_call(record)

    async def load_model_calls(self, run_id: str) -> Sequence[ModelCallRecord]:
        return await self.inner.load_model_calls(run_id)

    def for_node(self, node_id: str) -> list[str]:
        return [f"{kind}:{detail}" for kind, target, detail in self.events if target == node_id]


def diamond(source: str = "fake", branch: str = "fake", sink: str = "fake") -> Workflow:
    """A -> B, A -> C, B + C -> D."""
    return (
        WorkflowBuilder("diamond")
        .add_node("a", source)
        .add_node("b", branch, inputs={"upstream": "a"})
        .add_node("c", branch, inputs={"upstream": "a"})
        .add_node("d", sink, inputs={"left": "b", "right": "c"})
        .build()
    )


def registry_with(**agents: object) -> AgentRegistry:
    registry = AgentRegistry()
    for name, factory in agents.items():
        registry.add(name, factory)  # type: ignore[arg-type]
    return registry


# --- Phase 2 acceptance -------------------------------------------------------------


@pytest.mark.asyncio
async def test_diamond_runs_branches_concurrently_and_fans_in_both_outputs() -> None:
    probe = ConcurrencyProbe(parties=2)
    registry = registry_with(source=FakeAgent, branch=lambda: ProbeAgent(probe), sink=FakeAgent)
    store = InMemoryStateStore()
    executor = Executor(registry=registry, store=store)

    # The timeout is the assertion for "actually concurrent": serialised branches
    # deadlock on the barrier and surface here instead of hanging the suite.
    run = await asyncio.wait_for(
        executor.run(diamond(source="source", branch="branch", sink="sink"), run_id="r1"),
        timeout=5,
    )

    assert run.state is RunState.SUCCEEDED
    assert {record.state for record in run.nodes.values()} == {NodeState.SUCCESS}

    # B and C were in flight at the same moment...
    assert probe.peak == 2
    # ...and their execution intervals genuinely overlap.
    b_start, b_end = probe.intervals["b"]
    c_start, c_end = probe.intervals["c"]
    assert b_start < c_end
    assert c_start < b_end

    # D received both upstream outputs, under the local names it declared.
    assert await store.load_output("r1", "d") == {
        "node_id": "d",
        "attempt": 0,
        "inputs": {"left": {"node": "b"}, "right": {"node": "c"}},
    }


@pytest.mark.asyncio
async def test_every_node_of_the_diamond_ends_successful() -> None:
    store = InMemoryStateStore()
    run = await Executor(registry=registry_with(fake=FakeAgent), store=store).run(
        diamond(), run_id="r1"
    )

    assert [run.nodes[node_id].state for node_id in ("a", "b", "c", "d")] == [NodeState.SUCCESS] * 4


# --- output plumbing ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_source_node_receives_no_inputs() -> None:
    store = InMemoryStateStore()
    workflow = WorkflowBuilder("single").add_node("a", "fake").build()

    await Executor(registry=registry_with(fake=FakeAgent), store=store).run(workflow, run_id="r1")

    assert await store.load_output("r1", "a") == {"node_id": "a", "attempt": 0, "inputs": {}}


@pytest.mark.asyncio
async def test_outputs_flow_along_a_chain() -> None:
    store = InMemoryStateStore()
    workflow = (
        WorkflowBuilder("chain")
        .add_node("a", "fake")
        .add_node("b", "fake", inputs={"from_a": "a"})
        .add_node("c", "fake", inputs={"from_b": "b"})
        .build()
    )

    await Executor(registry=registry_with(fake=FakeAgent), store=store).run(workflow, run_id="r1")

    c_output = await store.load_output("r1", "c")
    assert c_output == {
        "node_id": "c",
        "attempt": 0,
        "inputs": {
            "from_b": {
                "node_id": "b",
                "attempt": 0,
                "inputs": {"from_a": {"node_id": "a", "attempt": 0, "inputs": {}}},
            }
        },
    }


@pytest.mark.asyncio
async def test_a_node_reading_one_upstream_twice_gets_it_under_both_names() -> None:
    store = InMemoryStateStore()
    workflow = (
        WorkflowBuilder("aliased")
        .add_node("a", "fake")
        .add_node("b", "fake", inputs={"first": "a", "second": "a"})
        .build()
    )

    await Executor(registry=registry_with(fake=FakeAgent), store=store).run(workflow, run_id="r1")

    output = await store.load_output("r1", "b")
    assert isinstance(output, dict)
    assert set(output["inputs"]) == {"first", "second"}  # type: ignore[arg-type, index]


# --- persistence and ordering -------------------------------------------------------


@pytest.mark.asyncio
async def test_the_spy_store_satisfies_the_protocol() -> None:
    assert isinstance(SpyStore(), StateStore)


@pytest.mark.asyncio
async def test_a_node_moves_pending_to_ready_to_running_to_success() -> None:
    store = SpyStore()
    workflow = WorkflowBuilder("single").add_node("a", "fake").build()

    await Executor(registry=registry_with(fake=FakeAgent), store=store).run(workflow, run_id="r1")

    assert store.for_node("a") == ["state:ready", "state:running", "output:", "state:success"]


@pytest.mark.asyncio
async def test_a_nodes_output_is_written_before_it_is_marked_success() -> None:
    # Otherwise a resumed run could skip a SUCCESS node whose output it cannot recover.
    store = SpyStore()

    await Executor(registry=registry_with(fake=FakeAgent), store=store).run(diamond(), run_id="r1")

    for node_id in ("a", "b", "c", "d"):
        events = store.for_node(node_id)
        assert events.index("output:") < events.index("state:success")


@pytest.mark.asyncio
async def test_the_definition_is_stored_before_the_run_that_executes_it() -> None:
    # A run record without a definition behind it is a run `resume` cannot continue.
    store = SpyStore()

    await Executor(registry=registry_with(fake=FakeAgent), store=store).run(diamond(), run_id="r1")

    assert store.events[0] == ("workflow", "r1", "diamond")
    assert store.events[1] == ("run", "r1", RunState.RUNNING)


@pytest.mark.asyncio
async def test_the_returned_record_is_the_one_in_the_store() -> None:
    store = InMemoryStateStore()

    run = await Executor(registry=registry_with(fake=FakeAgent), store=store).run(
        diamond(), run_id="r1"
    )

    assert run == await store.load_run("r1")


@pytest.mark.asyncio
async def test_timestamps_come_from_the_injected_clock() -> None:
    clock = ManualClock()
    store = InMemoryStateStore()
    workflow = WorkflowBuilder("single").add_node("a", "fake").build()

    run = await Executor(registry=registry_with(fake=FakeAgent), store=store, clock=clock).run(
        workflow, run_id="r1"
    )

    assert run.created_at == clock.now()
    assert run.nodes["a"].started_at == clock.now()
    assert run.nodes["a"].finished_at == clock.now()


# --- validation ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unregistered_agent_is_rejected_before_anything_runs() -> None:
    store = SpyStore()
    workflow = WorkflowBuilder("single").add_node("a", "nope").build()

    with pytest.raises(ValidationError, match="not registered"):
        await Executor(registry=registry_with(fake=FakeAgent), store=store).run(
            workflow, run_id="r1"
        )

    assert store.events == []


@pytest.mark.asyncio
async def test_an_invalid_graph_is_rejected_before_anything_runs() -> None:
    store = SpyStore()
    workflow = Workflow(
        name="dangling",
        nodes=(Node(id="a", agent="fake", depends_on=("ghost",)),),
    )

    with pytest.raises(ValidationError, match="ghost"):
        await Executor(registry=registry_with(fake=FakeAgent), store=store).run(
            workflow, run_id="r1"
        )

    assert store.events == []


# --- failure ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failing_node_is_recorded_and_does_not_hang_the_run() -> None:
    store = InMemoryStateStore()
    workflow = (
        WorkflowBuilder("failing")
        .add_node("a", "fake")
        .add_node("b", "boom", inputs={"x": "a"})
        .build()
    )

    run = await asyncio.wait_for(
        Executor(registry=registry_with(fake=FakeAgent, boom=FailingAgent), store=store).run(
            workflow, run_id="r1"
        ),
        timeout=5,
    )

    assert run.state is RunState.FAILED
    assert run.nodes["a"].state is NodeState.SUCCESS
    assert run.nodes["b"].state is NodeState.FAILED
    assert "RuntimeError" in (run.nodes["b"].error or "")


@pytest.mark.asyncio
async def test_a_node_blocked_by_a_failure_is_left_pending_for_now() -> None:
    # Phase 2 has no failure semantics: marking this SKIPPED is one of the three modes
    # Phase 4 makes configurable, so this pins the current behaviour rather than
    # inventing a policy early.
    store = InMemoryStateStore()
    workflow = (
        WorkflowBuilder("blocked")
        .add_node("a", "boom")
        .add_node("b", "fake", inputs={"x": "a"})
        .build()
    )

    run = await asyncio.wait_for(
        Executor(registry=registry_with(fake=FakeAgent, boom=FailingAgent), store=store).run(
            workflow, run_id="r1"
        ),
        timeout=5,
    )

    assert run.nodes["a"].state is NodeState.FAILED
    assert run.nodes["b"].state is NodeState.PENDING
    assert run.state is RunState.FAILED


@pytest.mark.asyncio
async def test_an_independent_branch_still_finishes_when_a_sibling_fails() -> None:
    store = InMemoryStateStore()
    workflow = WorkflowBuilder("siblings").add_node("ok", "fake").add_node("bad", "boom").build()

    run = await Executor(
        registry=registry_with(fake=FakeAgent, boom=FailingAgent), store=store
    ).run(workflow, run_id="r1")

    assert run.nodes["ok"].state is NodeState.SUCCESS
    assert run.nodes["bad"].state is NodeState.FAILED


# --- cancellation -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelling_a_run_stops_every_node_in_flight() -> None:
    entered = asyncio.Event()

    class Blocking:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            entered.set()
            await asyncio.Event().wait()  # never completes
            return None

    executor = Executor(registry=registry_with(block=Blocking), store=InMemoryStateStore())
    workflow = WorkflowBuilder("blocking").add_node("a", "block").build()

    run_task = asyncio.create_task(executor.run(workflow, run_id="r1"))
    await asyncio.wait_for(entered.wait(), timeout=5)
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    leaked = [task for task in asyncio.all_tasks() if task.get_name().startswith("dagent:")]
    assert leaked == []


def test_the_agent_protocol_is_satisfied_without_inheriting_anything() -> None:
    # Adding an agent requires no import from dagent (SPEC FR-4).
    agent: Agent = FakeAgent()
    assert agent is not None
