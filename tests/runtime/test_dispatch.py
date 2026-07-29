"""The dispatcher seam itself: the two implementations, at the edges of their contract.

The paths that matter here are the ones the ordinary run never takes — a duplicate result
arriving, a coordinator walking away, a node that ignored its cancellation. Each of them
exists because at-least-once delivery makes it possible, so each of them is asserted rather
than reasoned about.
"""

import asyncio
import contextlib

import pytest

from dagent.agents.fake import FakeAgent, HangingAgent
from dagent.graph.builder import WorkflowBuilder, build_node
from dagent.models.state import NodeState, RunState, RunStateRecord
from dagent.runtime.dispatch import Dispatcher, LocalDispatcher, QueueDispatcher
from dagent.runtime.expansion import RunGraph
from dagent.runtime.node import NodeRunner
from dagent.runtime.registry import AgentRegistry
from dagent.store.memory import InMemoryStateStore
from dagent.transport.base import WorkItem, WorkResult
from dagent.transport.memory import InMemoryWorkQueue

pytestmark = pytest.mark.asyncio

POLL = 0.01


def registry_with(**agents: object) -> AgentRegistry:
    registry = AgentRegistry()
    for name, factory in agents.items():
        registry.add(name, factory)  # type: ignore[arg-type]
    return registry


def graph_for(*node_ids: str) -> RunGraph:
    builder = WorkflowBuilder("g")
    for node_id in node_ids:
        builder.add_node(node_id, "fake")
    return RunGraph(builder.build(), known_agents=["fake"])


async def opened(store: InMemoryStateStore, graph: RunGraph, run_id: str = "r1") -> None:
    await store.save_workflow(run_id, graph.workflow)
    await store.checkpoint(
        RunStateRecord(run_id=run_id, workflow_name=graph.workflow.name, state=RunState.RUNNING)
    )


def local_for(store: InMemoryStateStore, **agents: object) -> LocalDispatcher:
    return LocalDispatcher(
        NodeRunner(registry=registry_with(**agents or {"fake": FakeAgent}), store=store),
        store=store,
    )


# --- both implementations -----------------------------------------------------------------


async def test_both_implementations_satisfy_the_protocol() -> None:
    store = InMemoryStateStore()

    assert isinstance(local_for(store), Dispatcher)
    assert isinstance(QueueDispatcher(InMemoryWorkQueue()), Dispatcher)


async def test_neither_reports_anything_outstanding_before_it_is_given_work() -> None:
    store = InMemoryStateStore()

    assert local_for(store).outstanding == frozenset()
    assert QueueDispatcher(InMemoryWorkQueue()).outstanding == frozenset()


# --- in process ---------------------------------------------------------------------------


async def test_a_local_dispatcher_has_nothing_to_recover() -> None:
    # Nothing in memory outlives the process that crashed, so the answer is always empty —
    # and it has to *be* empty rather than merely unimplemented, because the executor folds
    # whatever it gets in before doing anything else.
    store = InMemoryStateStore()
    graph = graph_for("a")
    await opened(store, graph)

    assert await local_for(store).start("r1", graph) == ()


async def test_a_local_node_that_swallowed_its_cancellation_keeps_its_real_outcome() -> None:
    # Reporting a node as cancelled when it actually finished would be a falsehood in the
    # store, which is worse than the word "cancelled" being slightly wrong.
    store = InMemoryStateStore()
    graph = graph_for("a")
    await opened(store, graph)

    class Stubborn:
        async def run(self, ctx: object) -> str:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0.05)
            return "finished anyway"

    dispatcher = local_for(store, fake=Stubborn)
    await dispatcher.start("r1", graph)
    await dispatcher.dispatch(graph.index.nodes["a"], attempt=0)
    await asyncio.sleep(0)

    (outcome,) = await dispatcher.halt()

    assert outcome.cancelled is False
    assert outcome.state is NodeState.SUCCESS


async def test_a_local_node_that_honours_its_cancellation_is_reported_cancelled() -> None:
    store = InMemoryStateStore()
    graph = graph_for("a")
    await opened(store, graph)
    hanging = HangingAgent()

    dispatcher = local_for(store, fake=lambda: hanging)
    await dispatcher.start("r1", graph)
    await dispatcher.dispatch(graph.index.nodes["a"], attempt=3)
    await hanging.entered.wait()

    (outcome,) = await dispatcher.halt()

    assert outcome.cancelled is True
    assert outcome.state is NodeState.FAILED
    # The attempt it was dispatched at, not zero: the record has to name the work that was
    # stopped, and a cancelled node's attempt is as real as a successful one's.
    assert outcome.attempt == 3
    assert dispatcher.outstanding == frozenset()


async def test_abandoning_a_local_run_stops_everything_and_reports_nothing() -> None:
    store = InMemoryStateStore()
    graph = graph_for("a")
    await opened(store, graph)
    hanging = HangingAgent()

    dispatcher = local_for(store, fake=lambda: hanging)
    await dispatcher.start("r1", graph)
    await dispatcher.dispatch(graph.index.nodes["a"], attempt=0)
    await hanging.entered.wait()

    await dispatcher.abandon()

    assert hanging.cancelled == 1
    assert dispatcher.outstanding == frozenset()


async def test_acknowledging_a_local_outcome_is_a_no_op() -> None:
    store = InMemoryStateStore()
    graph = graph_for("a")
    await opened(store, graph)
    dispatcher = local_for(store)
    await dispatcher.start("r1", graph)

    await dispatcher.acknowledge([])


# --- across a queue -----------------------------------------------------------------------


async def test_a_duplicate_result_is_acknowledged_and_dropped_rather_than_folded_twice() -> None:
    # At-least-once delivery makes this ordinary rather than exceptional: the same node can
    # be reported twice by two workers that both ran it. Folding the second one would
    # double-count a completion the loop has already acted on.
    queue = InMemoryWorkQueue()
    dispatcher = QueueDispatcher(queue, poll_timeout_s=POLL)
    await dispatcher.start("r1", graph_for("a"))
    await dispatcher.dispatch(build_node("a", "fake"), attempt=0)

    await queue.submit(WorkItem("r1", "a", 0))
    claimed = await queue.claim(consumer="w1", timeout_s=POLL)
    assert claimed is not None
    await queue.complete(claimed, WorkResult("r1", "a", 0, NodeState.SUCCESS))

    first = await dispatcher.settle()
    assert [outcome.node_id for outcome in first] == ["a"]
    await dispatcher.acknowledge(first)

    # A second report of the same node, from a worker that also ran it.
    await queue.submit(WorkItem("r1", "a", 0))
    again = await queue.claim(consumer="w2", timeout_s=POLL)
    assert again is not None
    await queue.complete(again, WorkResult("r1", "a", 0, NodeState.SUCCESS))

    # `settle` blocks until something relevant arrives, and nothing ever will — the point is
    # that the duplicate was consumed and retired rather than returned.
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(dispatcher.settle(), timeout=0.2)
    assert await queue.collect("r1", timeout_s=POLL) == ()


async def test_halting_a_queued_run_waits_for_the_nodes_already_out_there() -> None:
    # It cannot cancel them: there is no handle to cancel a coroutine in another process.
    queue = InMemoryWorkQueue()
    dispatcher = QueueDispatcher(queue, poll_timeout_s=POLL)
    await dispatcher.start("r1", graph_for("a", "b"))
    await dispatcher.dispatch(build_node("a", "fake"), attempt=0)
    await dispatcher.dispatch(build_node("b", "fake"), attempt=0)
    assert dispatcher.outstanding == {"a", "b"}

    async def answer() -> None:
        for _ in range(2):
            item = await queue.claim(consumer="w1", timeout_s=1)
            assert item is not None
            await queue.complete(
                item, WorkResult(item.run_id, item.node_id, item.attempt, NodeState.SUCCESS)
            )

    worker = asyncio.create_task(answer())
    outcomes = await dispatcher.halt()
    await worker

    assert {outcome.node_id for outcome in outcomes} == {"a", "b"}
    assert all(not outcome.cancelled for outcome in outcomes)
    assert dispatcher.outstanding == frozenset()


async def test_halting_with_nothing_outstanding_is_immediate() -> None:
    dispatcher = QueueDispatcher(InMemoryWorkQueue(), poll_timeout_s=POLL)
    await dispatcher.start("r1", graph_for("a"))

    assert await dispatcher.halt() == ()


async def test_abandoning_a_queued_run_leaves_the_work_alone() -> None:
    # The coordinator is leaving; the workers are not. Whatever they finish lands in the
    # store, and a resumed coordinator reads it from there.
    queue = InMemoryWorkQueue()
    dispatcher = QueueDispatcher(queue, poll_timeout_s=POLL)
    await dispatcher.start("r1", graph_for("a"))
    await dispatcher.dispatch(build_node("a", "fake"), attempt=0)

    await dispatcher.abandon()

    assert dispatcher.outstanding == frozenset()
    assert [item.node_id for item in queue.backlog] == ["a"]


async def test_starting_a_second_run_forgets_the_first() -> None:
    queue = InMemoryWorkQueue()
    dispatcher = QueueDispatcher(queue, poll_timeout_s=POLL)
    await dispatcher.start("r1", graph_for("a"))
    await dispatcher.dispatch(build_node("a", "fake"), attempt=0)

    await dispatcher.start("r2", graph_for("a"))

    assert dispatcher.outstanding == frozenset()
