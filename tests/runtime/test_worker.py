"""The stateless worker: claim, run, report, acknowledge — and what happens when it dies.

Driven one `step()` at a time rather than through `run_forever`, so each property is
asserted at the point it holds instead of being inferred from a race. The queue is the
in-memory one, whose consumer-group semantics `tests/transport/test_queue_conformance.py`
holds to the same contract Redis is held to.
"""

import asyncio

import pytest

from dagent.agents.fake import FailingAgent, FakeAgent, SideEffectAgent
from dagent.errors import StoreError
from dagent.graph.builder import WorkflowBuilder, build_node
from dagent.models.state import NodeOutput, NodeState, RunState, RunStateRecord
from dagent.models.workflow import Workflow
from dagent.runtime.agent import AgentContext
from dagent.runtime.registry import AgentRegistry
from dagent.runtime.worker import Worker
from dagent.store.base import StateStore
from dagent.store.memory import InMemoryStateStore
from dagent.transport.base import WorkItem
from dagent.transport.memory import InMemoryWorkQueue

pytestmark = pytest.mark.asyncio


def registry_with(**agents: object) -> AgentRegistry:
    registry = AgentRegistry()
    for name, factory in agents.items():
        registry.add(name, factory)  # type: ignore[arg-type]
    return registry


def one_node(agent: str = "fake") -> Workflow:
    return WorkflowBuilder("solo").add_node("a", agent).build()


async def opened(store: StateStore, workflow: Workflow, run_id: str = "r1") -> None:
    """Put a run in the store the way the coordinator would before dispatching anything."""
    await store.save_workflow(run_id, workflow)
    await store.checkpoint(
        RunStateRecord(run_id=run_id, workflow_name=workflow.name, state=RunState.RUNNING)
    )


def worker_for(
    store: StateStore, queue: InMemoryWorkQueue, *, name: str = "w1", **kwargs: object
) -> Worker:
    return Worker(
        name=name,
        registry=kwargs.pop("registry", registry_with(fake=FakeAgent)),  # type: ignore[arg-type]
        store=store,
        queue=queue,
        claim_timeout_s=0.02,
        **kwargs,  # type: ignore[arg-type]
    )


# --- the ordinary life of one node --------------------------------------------------------


async def test_a_worker_runs_the_node_it_claimed_and_records_the_verdict() -> None:
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, one_node())
    await queue.submit(WorkItem("r1", "a", 0))

    assert await worker_for(store, queue).step() is not None

    run = await store.load_run("r1")
    assert run.nodes["a"].state is NodeState.SUCCESS
    assert await store.load_output("r1", "a") == {"node_id": "a", "attempt": 0, "inputs": {}}


async def test_the_verdict_is_published_where_the_coordinator_will_find_it() -> None:
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, one_node())
    await queue.follow("r1")
    await queue.submit(WorkItem("r1", "a", 0))

    await worker_for(store, queue).step()

    (result,) = await queue.collect("r1", timeout_s=0.02)
    assert (result.node_id, result.attempt, result.state) == ("a", 0, NodeState.SUCCESS)


async def test_the_item_is_acknowledged_only_after_the_result_is_published() -> None:
    # Nothing left pending means nobody will run this node again — which must not be true
    # until its outcome is somewhere the coordinator can read.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, one_node())
    await queue.submit(WorkItem("r1", "a", 0))

    await worker_for(store, queue).step()

    assert queue.pending == ()
    assert len(await queue.collect("r1", timeout_s=0.02)) == 1


async def test_a_worker_with_nothing_to_do_says_so_rather_than_blocking() -> None:
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()

    assert await worker_for(store, queue).step() is None


async def test_a_worker_runs_the_attempt_it_was_told_to() -> None:
    # The attempt travels in the message, because it is half of the idempotency key. A
    # worker that started counting from zero would hand the agent the wrong key.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, one_node())
    await queue.submit(WorkItem("r1", "a", 4))

    await worker_for(store, queue).step()

    assert (await store.load_output("r1", "a"))["attempt"] == 4  # type: ignore[index,call-overload]
    assert (await store.load_run("r1")).nodes["a"].attempt == 4


async def test_a_failing_node_is_reported_as_failed_rather_than_raised() -> None:
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, one_node("boom"))
    await queue.follow("r1")
    await queue.submit(WorkItem("r1", "a", 0))

    await worker_for(store, queue, registry=registry_with(boom=FailingAgent)).step()

    (result,) = await queue.collect("r1", timeout_s=0.02)
    assert result.state is NodeState.FAILED
    assert "RuntimeError" in (await store.load_run("r1")).nodes["a"].error  # type: ignore[operator]


# --- the graph the worker never owns ------------------------------------------------------


async def test_a_worker_reads_the_definition_from_the_store_not_from_the_message() -> None:
    # The message carries the idempotency key and nothing else. If it carried the node,
    # there would be two sources of truth for what a node is, and they would disagree the
    # first time a planner expanded anything.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, WorkflowBuilder("solo").add_node("a", "fake", params={"k": "v"}).build())
    await queue.submit(WorkItem("r1", "a", 0))

    class Echo:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            return dict(ctx.params)

    await worker_for(store, queue, registry=registry_with(fake=Echo)).step()

    assert await store.load_output("r1", "a") == {"k": "v"}


async def test_a_node_the_worker_has_not_heard_of_makes_it_reload_the_graph() -> None:
    # How a worker keeps up with a graph that grew. The coordinator persists an expansion
    # before dispatching anything it added, so a node this worker does not know is always
    # a stale cache and never a missing node.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, one_node())
    await queue.submit(WorkItem("r1", "a", 0))
    worker = worker_for(store, queue)
    await worker.step()  # caches the one-node graph

    grown = WorkflowBuilder("solo").add_node("a", "fake").add_node("b", "fake").build()
    await store.save_workflow("r1", grown)
    await queue.submit(WorkItem("r1", "b", 0))

    await worker.step()

    assert (await store.load_run("r1")).nodes["b"].state is NodeState.SUCCESS


async def test_a_node_that_is_genuinely_not_in_the_graph_fails_that_node_only() -> None:
    # Reported rather than raised: an unacknowledged item would be handed to the next
    # worker to fail in exactly the same way, forever.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, one_node())
    await queue.follow("r1")
    await queue.submit(WorkItem("r1", "ghost", 0))

    await worker_for(store, queue).step()

    (result,) = await queue.collect("r1", timeout_s=0.02)
    assert result.state is NodeState.FAILED
    assert "not in the stored definition" in (await store.load_run("r1")).nodes["ghost"].error  # type: ignore[operator]
    assert queue.pending == ()


async def test_a_planner_reports_its_expansion_instead_of_merging_it() -> None:
    # The worker holds a copy of the workflow and could have merged this itself. It must
    # not: another worker's planner may have grown the graph since that copy was loaded,
    # so only the coordinator is looking at a graph nobody else is editing.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, one_node("planner"))
    await queue.follow("r1")
    await queue.submit(WorkItem("r1", "a", 0))

    class Planner:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            ctx.expand(build_node("grown", "fake", depends_on=[ctx.node_id]))
            return "planned"

    await worker_for(store, queue, registry=registry_with(planner=Planner, fake=FakeAgent)).step()

    (result,) = await queue.collect("r1", timeout_s=0.02)
    assert [node.id for node in result.expansion] == ["grown"]
    # Unchanged: the definition in the store is still the coordinator's to write.
    assert [node.id for node in (await store.load_workflow("r1")).nodes] == ["a"]


# --- a worker that dies ------------------------------------------------------------------


async def test_a_node_a_worker_never_reported_is_taken_over_by_another() -> None:
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, one_node())
    await queue.submit(WorkItem("r1", "a", 0))

    # A worker that claims and then dies: no `complete`, so the entry stays pending.
    dead = worker_for(store, queue, name="dead")
    claimed = await queue.claim(consumer="dead", timeout_s=0.02)
    assert claimed is not None
    del dead

    alive = worker_for(store, queue, name="alive", reclaim_after_s=0)
    assert await alive.step() is not None

    assert (await store.load_run("r1")).nodes["a"].state is NodeState.SUCCESS


async def test_a_taken_over_node_runs_under_the_key_the_first_worker_used() -> None:
    # The acceptance criterion, in miniature: at-least-once delivery is safe *because* the
    # redelivered item names the same attempt, so an agent that deduplicates on the key
    # commits once however many times the node runs.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, one_node("effect"))
    ledger: dict[str, NodeOutput] = {}
    agent = SideEffectAgent(ledger)
    await queue.submit(WorkItem("r1", "a", 0))

    # First worker: claims, commits the effect, then dies before acknowledging anything.
    first = worker_for(store, queue, name="dead", registry=registry_with(effect=lambda: agent))
    claimed = await queue.claim(consumer="dead", timeout_s=0.02)
    assert claimed is not None
    await first._execute(claimed)  # ran the node; never got to `complete`
    assert agent.commits == 1

    second = worker_for(
        store, queue, name="alive", registry=registry_with(effect=lambda: agent), reclaim_after_s=0
    )
    assert await second.step() is not None

    assert agent.commits == 1
    assert list(ledger) == ["r1:a:0"]


# --- the loop ---------------------------------------------------------------------------


async def test_run_forever_stops_when_it_is_told_to() -> None:
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    stop = asyncio.Event()
    worker = worker_for(store, queue)

    running = asyncio.create_task(worker.run_forever(stop=stop))
    await asyncio.sleep(0)
    stop.set()

    assert await asyncio.wait_for(running, timeout=1) == 0


async def test_run_forever_counts_the_nodes_it_ran_and_honours_a_limit() -> None:
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, WorkflowBuilder("two").add_node("a", "fake").add_node("b", "fake").build())
    await queue.submit(WorkItem("r1", "a", 0))
    await queue.submit(WorkItem("r1", "b", 0))

    assert await worker_for(store, queue).run_forever(limit=2) == 2


async def test_a_step_that_fails_leaves_its_item_for_somebody_else() -> None:
    # Nothing is acknowledged, so the delivery is still pending and will be reclaimed. A
    # worker that swallowed the item along with the error would lose the node silently.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await queue.submit(WorkItem("nonexistent-run", "a", 0))

    with pytest.raises(StoreError):
        await worker_for(store, queue).step()

    assert [item.node_id for item in queue.pending] == ["a"]


async def test_a_store_failure_does_not_kill_the_worker() -> None:
    # Exiting over one bad message would turn a recoverable delivery into a lost process,
    # so the loop logs it and carries on to the next node.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, one_node())
    await queue.submit(WorkItem("nonexistent-run", "a", 0))
    await queue.submit(WorkItem("r1", "a", 0))

    handled = await worker_for(store, queue).run_forever(limit=1)

    assert handled == 1
    assert (await store.load_run("r1")).nodes["a"].state is NodeState.SUCCESS


async def test_a_batch_of_reclaimed_work_is_worked_through_one_item_at_a_time() -> None:
    # `reclaim` answers with everything that went stale, which can be more than one node.
    # Buffering the rest is what stops the extras from being claimed by this worker and then
    # sitting untouched until they go stale a second time.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await opened(store, WorkflowBuilder("two").add_node("a", "fake").add_node("b", "fake").build())
    for node_id in ("a", "b"):
        await queue.submit(WorkItem("r1", node_id, 0))
        assert await queue.claim(consumer="dead", timeout_s=0.02) is not None

    worker = worker_for(store, queue, name="alive", reclaim_after_s=0)
    first, second = await worker.step(), await worker.step()

    assert {first.node_id, second.node_id} == {"a", "b"}  # type: ignore[union-attr]
    run = await store.load_run("r1")
    assert [run.nodes[node_id].state for node_id in ("a", "b")] == [NodeState.SUCCESS] * 2
    assert queue.pending == ()


async def test_the_definition_cache_forgets_the_run_it_touched_longest_ago(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A worker is long-lived and sees every run in the pool, so the cache has to be bounded.
    monkeypatch.setattr("dagent.runtime.worker._CACHED_RUNS", 2)
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    worker = worker_for(store, queue)
    for run_id in ("r1", "r2", "r3"):
        await opened(store, one_node(), run_id=run_id)
        await queue.submit(WorkItem(run_id, "a", 0))
        assert await worker.step() is not None

    assert list(worker._definitions) == ["r2", "r3"]
