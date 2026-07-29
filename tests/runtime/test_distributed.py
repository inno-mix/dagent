"""Phase 8 acceptance: the research workflow, run across a pool of workers.

The criterion is that the showcase workflow — a planner that decides at run time how many
researchers there should be, plus a synthesizer that fans them back in — runs across two or
more workers, that killing one mid-run corrupts nothing and duplicates no side effect, and
that the core did not have to change to allow it.

The last part is what these tests are really for, so they are deliberately written against
the *same* agents, the *same* workflow, and the *same* `Executor` the single-process tests
use. Nothing here is a distributed variant of anything. The only differences are the
dispatcher handed to the executor and the workers standing by to take what it hands out.

Workers are asyncio tasks rather than OS processes, for the reason every test in this
repository is offline: a suite that needs to fork is a suite that gets skipped. What makes
the substitution honest is that they talk to each other only through the store and the
queue, exactly as processes do — no shared graph, no shared state map, no shared registry
instance — and that the in-memory queue is held to Redis' consumer-group semantics by
``tests/transport/test_queue_conformance.py``. The real two-process run is verified by hand
against Redis and Postgres; see the README.
"""

import asyncio

import pytest

from dagent.agents.fake import ConstantAgent, FailingAgent, FakeAgent, SideEffectAgent
from dagent.agents.planner import PlannerAgent
from dagent.agents.researcher import ResearcherAgent
from dagent.agents.synthesizer import SynthesizerAgent
from dagent.graph.builder import WorkflowBuilder, build_node
from dagent.models.model_call import ModelRequest
from dagent.models.state import (
    NodeOutput,
    NodeState,
    NodeStateRecord,
    RunState,
    RunStateRecord,
)
from dagent.models.workflow import Workflow
from dagent.policy.run import FailureMode, RunPolicy
from dagent.runtime.agent import AgentContext
from dagent.runtime.dispatch import QueueDispatcher
from dagent.runtime.executor import Executor
from dagent.runtime.model import ModelClient, StubModelClient
from dagent.runtime.registry import AgentRegistry
from dagent.runtime.worker import Worker
from dagent.store.base import StateStore
from dagent.store.memory import InMemoryStateStore
from dagent.transport.base import WorkItem
from dagent.transport.memory import InMemoryWorkQueue

pytestmark = pytest.mark.asyncio

POLL = 0.01
"""How long a coordinator or worker blocks on an empty channel. Short, because the tests
are the only thing waiting."""

SUBTOPICS = ("instruction tuning", "RLHF", "synthetic data")


def registry_with(**agents: object) -> AgentRegistry:
    registry = AgentRegistry()
    for name, factory in agents.items():
        registry.add(name, factory)  # type: ignore[arg-type]
    return registry


def research_agents() -> AgentRegistry:
    """The four agents `examples/research_dynamic.yaml` names — the real ones."""
    return registry_with(
        constant=ConstantAgent,
        planner=PlannerAgent,
        researcher=ResearcherAgent,
        synthesizer=SynthesizerAgent,
    )


class NeverRuns:
    """An agent that cannot be constructed, let alone run.

    Given to the coordinator so that a node executed in the coordinator's own process is a
    loud failure rather than a passing test. Validation only needs the agent *names*, which
    is why substituting this for the real registry proves the work really left.
    """

    def __init__(self) -> None:
        raise AssertionError("the coordinator ran an agent; the work should have gone to a worker")


def coordinator_registry() -> AgentRegistry:
    return registry_with(
        constant=NeverRuns, planner=NeverRuns, researcher=NeverRuns, synthesizer=NeverRuns
    )


def research(subtopics: int = 3) -> Workflow:
    """`examples/research_dynamic.yaml`: two nodes written down, the rest decided at run time."""
    return (
        WorkflowBuilder("research-dynamic")
        .add_node("question", "constant", params={"value": "How do LLMs follow instructions?"})
        .add_node(
            "plan", "planner", inputs={"question": "question"}, params={"subtopics": subtopics}
        )
        .build()
    )


def stub_model() -> ModelClient:
    """A scripted provider: subtopics for the planner, a short answer for everyone else."""

    def reply(request: ModelRequest) -> str:
        if "subtopics, one per line" in request.prompt:
            return "\n".join(SUBTOPICS)
        return f"[answer to {request.prompt.splitlines()[0][:40]}]"

    return StubModelClient(reply)


async def run_across(
    workflow: Workflow | None,
    *,
    registry: AgentRegistry,
    store: StateStore,
    queue: InMemoryWorkQueue,
    workers: int = 2,
    model: ModelClient | None = None,
    policy: RunPolicy | None = None,
    run_id: str = "r1",
    coordinator: AgentRegistry | None = None,
    reclaim_after_s: float = 5.0,
    resume: bool = False,
) -> tuple[RunStateRecord, list[int]]:
    """Drive one run as a coordinator with a pool of workers, and report who did what.

    Returns the final run record and how many nodes each worker handled — the second half
    being how a test can tell "it ran across the pool" from "one worker did everything".
    """
    stop = asyncio.Event()
    pool = [
        asyncio.create_task(
            Worker(
                name=f"w{index}",
                registry=registry,
                store=store,
                queue=queue,
                model=model,
                policy=policy,
                claim_timeout_s=POLL,
                reclaim_after_s=reclaim_after_s,
            ).run_forever(stop=stop),
            name=f"worker-{index}",
        )
        for index in range(workers)
    ]
    executor = Executor(
        registry=coordinator if coordinator is not None else registry,
        store=store,
        model=model,
        policy=policy,
        dispatcher=QueueDispatcher(queue, poll_timeout_s=POLL),
    )
    try:
        record = (
            await executor.resume(run_id) if resume else await executor.run(workflow, run_id=run_id)  # type: ignore[arg-type]
        )
    finally:
        stop.set()
        handled = await asyncio.gather(*pool)
    return record, list(handled)


# --- the acceptance criterion -------------------------------------------------------------


async def test_the_research_workflow_runs_across_a_pool_of_workers() -> None:
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()

    record, handled = await run_across(
        research(),
        registry=research_agents(),
        coordinator=coordinator_registry(),
        store=store,
        queue=queue,
        model=stub_model(),
    )

    assert record.state is RunState.SUCCEEDED
    # Two written down, three the planner decided on, one fan-in it emitted alongside them.
    assert sorted(record.nodes) == [
        "plan",
        "plan.research_0",
        "plan.research_1",
        "plan.research_2",
        "plan.synthesis",
        "question",
    ]
    assert all(node.state is NodeState.SUCCESS for node in record.nodes.values())
    assert sum(handled) == 6


async def test_the_number_of_researchers_is_still_decided_at_run_time() -> None:
    # The shape came from the model, not from the file, and distributing the execution did
    # not move that decision anywhere.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()

    record, _ = await run_across(
        research(subtopics=2),
        registry=research_agents(),
        store=store,
        queue=queue,
        model=stub_model(),
    )

    researchers = [node_id for node_id in record.nodes if ".research_" in node_id]
    assert len(researchers) == 2
    assert (await store.load_output("r1", "plan"))["subtopics"] == list(SUBTOPICS[:2])  # type: ignore[index,call-overload]


async def test_more_than_one_worker_actually_shares_the_work() -> None:
    # The nodes have to take *time* for this to mean anything. With instantaneous agents a
    # single worker drains the whole backlog before the event loop ever schedules a second
    # one — which is not a scheduling bug, it is what "first come, first served" means when
    # the work is free. Real nodes wait on a model, so these do too, and then the pool
    # overlaps for the same reason a real one does.
    running = 0
    peak = 0

    class Slow:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            try:
                await asyncio.sleep(0.02)
            finally:
                running -= 1
            return ctx.node_id

    fan_out = WorkflowBuilder("wide").add_node("source", "slow")
    for index in range(4):
        fan_out.add_node(f"leaf{index}", "slow", depends_on=["source"])

    _, handled = await run_across(
        fan_out.build(),
        registry=registry_with(slow=Slow),
        store=InMemoryStateStore(),
        queue=InMemoryWorkQueue(),
        workers=3,
    )

    assert sum(handled) == 5
    assert sum(1 for count in handled if count) >= 2
    # And they really were in flight together, rather than merely being counted by two.
    assert peak >= 2


async def test_a_distributed_run_produces_exactly_what_a_single_process_run_produces() -> None:
    # The whole claim of DR-12 in one assertion: same definition, same agents, same
    # scheduler, different transport, identical result.
    here, there = InMemoryStateStore(), InMemoryStateStore()

    single = await Executor(registry=research_agents(), store=here, model=stub_model()).run(
        research(), run_id="r1"
    )
    distributed, _ = await run_across(
        research(),
        registry=research_agents(),
        store=there,
        queue=InMemoryWorkQueue(),
        model=stub_model(),
    )

    assert {name: node.state for name, node in single.nodes.items()} == {
        name: node.state for name, node in distributed.nodes.items()
    }
    for node_id in single.nodes:
        assert await here.load_output("r1", node_id) == await there.load_output("r1", node_id)


async def test_killing_a_worker_mid_run_neither_corrupts_state_nor_repeats_the_effect() -> None:
    # A worker claims a node, commits its side effect, and dies without reporting. Nobody
    # on this side of that can know whether the effect landed, so the node comes back under
    # the *same* idempotency key and the agent's own check is what makes the second
    # execution free. That is DR-4 being cashed in.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    ledger: dict[str, NodeOutput] = {}
    effect = SideEffectAgent(ledger)
    registry = registry_with(effect=lambda: effect, fake=FakeAgent)
    workflow = (
        WorkflowBuilder("chain")
        .add_node("a", "effect")
        .add_node("b", "fake", inputs={"x": "a"})
        .build()
    )

    # No pool yet, so the doomed worker is the only thing claiming and cannot lose the
    # race. Racing it against the survivors would let this test pass without the takeover
    # ever happening.
    coordinating = asyncio.create_task(
        Executor(
            registry=registry,
            store=store,
            dispatcher=QueueDispatcher(queue, poll_timeout_s=POLL),
        ).run(workflow, run_id="r1")
    )
    claimed = None
    while claimed is None:
        claimed = await queue.claim(consumer="doomed", timeout_s=POLL)

    # SIGKILL, in effect: the node ran and its effect landed, and then the process was gone
    # before it could say so. Nothing on this side can tell that from "never started".
    await Worker(name="doomed", registry=registry, store=store, queue=queue)._execute(claimed)
    assert effect.commits == 1
    assert [item.node_id for item in queue.pending] == ["a"]

    stop = asyncio.Event()
    survivor = asyncio.create_task(
        Worker(
            name="survivor",
            registry=registry,
            store=store,
            queue=queue,
            claim_timeout_s=POLL,
            reclaim_after_s=0,
        ).run_forever(stop=stop)
    )
    try:
        record = await coordinating
    finally:
        stop.set()
        handled = [await survivor]

    assert record.state is RunState.SUCCEEDED
    assert [node.state for node in record.nodes.values()] == [NodeState.SUCCESS] * 2
    # `a` ran twice — once in the worker that died, once in the one that took it over — and
    # committed once, under one key rather than two. That is the whole of "at-least-once is
    # safe because execution is idempotent".
    assert effect.commits == 1
    assert list(ledger) == ["r1:a:0"]
    assert handled == [2]
    assert queue.pending == ()


# --- the graph still has exactly one owner ------------------------------------------------


async def test_the_coordinator_persists_a_grown_graph_before_dispatching_into_it() -> None:
    # A worker looks its node up in the store, so a node dispatched before the augmented
    # definition was written would be a node no worker could find.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()

    await run_across(
        research(),
        registry=research_agents(),
        coordinator=coordinator_registry(),
        store=store,
        queue=queue,
        model=stub_model(),
    )

    stored = await store.load_workflow("r1")
    assert [node.id for node in stored.nodes] == [
        "question",
        "plan",
        "plan.research_0",
        "plan.research_1",
        "plan.research_2",
        "plan.synthesis",
    ]


async def test_an_expansion_the_coordinator_rejects_fails_its_node_and_nothing_else() -> None:
    # The worker cannot judge this — its copy of the graph may be stale — so the verdict
    # comes from the coordinator, one hop later than in a single-process run. What must not
    # happen is a deadlock: the run ends, and the node that asked carries the failure.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()

    class Cyclic:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            # The cycle has to be *among the added nodes*: expansion is append-only, so an
            # existing node can never gain a dependency and can never be drawn into a loop.
            # Two generated nodes feeding each other is the shape a real planner bug takes.
            ctx.expand(
                build_node("loop_a", "fake", inputs={"x": "loop_b"}),
                build_node("loop_b", "fake", inputs={"y": "loop_a"}),
            )
            return None

    workflow = (
        WorkflowBuilder("bad")
        .add_node("plan", "cyclic")
        .add_node("sibling", "fake", inputs={"x": "plan"})
        .build()
    )

    record, _ = await run_across(
        workflow,
        registry=registry_with(cyclic=Cyclic, fake=FakeAgent),
        store=store,
        queue=queue,
    )

    assert record.state is RunState.FAILED
    assert record.nodes["plan"].state is NodeState.FAILED
    assert "ValidationError" in (record.nodes["plan"].error or "")
    assert {node.id for node in (await store.load_workflow("r1")).nodes} == {"plan", "sibling"}


async def test_a_worker_never_writes_the_definition() -> None:
    written: list[str] = []

    class Watching(InMemoryStateStore):
        async def save_workflow(self, run_id: str, workflow: Workflow) -> None:
            written.append(f"{len(workflow.nodes)} nodes")
            await super().save_workflow(run_id, workflow)

    watched = Watching()

    await run_across(
        research(subtopics=1),
        registry=research_agents(),
        coordinator=coordinator_registry(),
        store=watched,
        queue=InMemoryWorkQueue(),
        model=stub_model(),
    )

    # Twice, both from the coordinator: the definition as submitted, then as grown.
    assert written == ["2 nodes", "4 nodes"]


# --- policy, unchanged --------------------------------------------------------------------


async def test_skip_downstream_still_marks_the_blast_radius() -> None:
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    workflow = (
        WorkflowBuilder("failing")
        .add_node("bad", "boom")
        .add_node("blocked", "fake", inputs={"x": "bad"})
        .build()
    )

    record, _ = await run_across(
        workflow,
        registry=registry_with(boom=FailingAgent, fake=FakeAgent),
        store=store,
        queue=queue,
        policy=RunPolicy(failure_mode=FailureMode.SKIP_DOWNSTREAM),
    )

    assert record.nodes["bad"].state is NodeState.FAILED
    assert record.nodes["blocked"].state is NodeState.SKIPPED


async def test_fail_fast_stops_dispatching_and_reports_what_really_happened() -> None:
    # A coroutine in another process cannot be cancelled, so `fail_fast` degrades from
    # "cancel the siblings" to "start nothing further and let the siblings land". The
    # sibling therefore reports SUCCESS, because that is what it did — recording it as
    # cancelled would put a falsehood in the store to preserve a word.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    workflow = (
        WorkflowBuilder("wide")
        .add_node("bad", "boom")
        .add_node("sibling", "fake")
        .add_node("later", "fake", inputs={"x": "sibling"})
        .build()
    )

    record, _ = await run_across(
        workflow,
        registry=registry_with(boom=FailingAgent, fake=FakeAgent),
        store=store,
        queue=queue,
        workers=1,
        policy=RunPolicy(failure_mode=FailureMode.FAIL_FAST),
    )

    assert record.state is RunState.FAILED
    assert record.nodes["bad"].state is NodeState.FAILED
    assert record.nodes["sibling"].state is NodeState.SUCCESS
    assert record.nodes["later"].state is NodeState.PENDING


async def test_a_node_policy_from_the_definition_is_honoured_by_the_worker() -> None:
    # Node-level policy lives in the frozen definition, which the worker reads from the
    # store — so retries work in a process that never saw the workflow file.
    from dagent.agents.fake import FlakyAgent
    from dagent.models.workflow import Policy
    from dagent.policy.retry import Backoff, no_jitter
    from dagent.runtime.clock import ManualClock

    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    workflow = (
        WorkflowBuilder("flaky")
        .add_node("a", "flaky", policy=Policy(max_attempts=3, backoff_initial_s=1.0))
        .build()
    )

    stop = asyncio.Event()
    registry = registry_with(flaky=lambda: FlakyAgent(fail_until_attempt=2))
    policy = RunPolicy(backoff=Backoff(jitter=no_jitter))
    pool = asyncio.create_task(
        Worker(
            name="w0",
            registry=registry,
            store=store,
            queue=queue,
            clock=ManualClock(),
            policy=policy,
            claim_timeout_s=POLL,
        ).run_forever(stop=stop)
    )
    try:
        record = await Executor(
            registry=registry,
            store=store,
            policy=policy,
            dispatcher=QueueDispatcher(queue, poll_timeout_s=POLL),
        ).run(workflow, run_id="r1")
    finally:
        stop.set()
        await pool

    assert record.state is RunState.SUCCEEDED
    assert record.nodes["a"].attempt == 2


# --- resume, across the queue -------------------------------------------------------------


async def test_a_distributed_run_can_be_resumed_by_a_new_coordinator() -> None:
    # The first coordinator dies with a node still out. The second one reloads the run from
    # the store, re-dispatches what was interrupted at the same attempt, and finishes.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    registry = registry_with(fake=FakeAgent)
    workflow = (
        WorkflowBuilder("chain")
        .add_node("a", "fake")
        .add_node("b", "fake", inputs={"x": "a"})
        .build()
    )

    # Stand in for a coordinator that opened the run and then vanished before dispatching
    # anything: the definition and the run record are there, and no node has a state.
    await store.save_workflow("r1", workflow)
    await store.checkpoint(
        RunStateRecord(run_id="r1", workflow_name="chain", state=RunState.RUNNING)
    )

    record, handled = await run_across(
        None, registry=registry, store=store, queue=queue, workers=1, resume=True
    )

    assert record.state is RunState.SUCCEEDED
    assert handled == [2]


async def test_a_result_the_coordinator_never_settled_is_delivered_again() -> None:
    # The window a coordinator crash opens: a planner's completion read but not acted on.
    # Because results are acknowledged only after they have been folded in, the report
    # comes back rather than the expansion being lost.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    await store.save_workflow("r1", research(subtopics=1))
    await store.checkpoint(
        RunStateRecord(run_id="r1", workflow_name="research-dynamic", state=RunState.RUNNING)
    )
    await queue.follow("r1")

    # A worker runs the planner; the result is published and never settled.
    registry = research_agents()
    worker = Worker(name="w0", registry=registry, store=store, queue=queue, model=stub_model())
    await queue.submit(WorkItem("r1", "question", 0))
    await worker.step()
    await queue.submit(WorkItem("r1", "plan", 0))
    await worker.step()
    unsettled = {result.node_id: result for result in await queue.collect("r1", timeout_s=POLL)}
    assert set(unsettled) == {"question", "plan"}
    assert unsettled["plan"].expansion  # the fan-out nobody has merged yet

    record, _ = await run_across(
        None, registry=registry, store=store, queue=queue, model=stub_model(), resume=True
    )

    assert record.state is RunState.SUCCEEDED
    assert "plan.synthesis" in record.nodes


async def test_a_failure_recovered_on_resume_still_drives_the_failure_mode() -> None:
    # A node that failed while nobody was listening. The report arrives on resume, and the
    # blast radius has to be marked from there — the failure mode cannot only apply to
    # failures this coordinator happened to dispatch itself.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    registry = registry_with(boom=FailingAgent, fake=FakeAgent)
    workflow = (
        WorkflowBuilder("failing")
        .add_node("bad", "boom")
        .add_node("blocked", "fake", inputs={"x": "bad"})
        .build()
    )
    await store.save_workflow("r1", workflow)
    await store.checkpoint(
        RunStateRecord(run_id="r1", workflow_name="failing", state=RunState.RUNNING)
    )
    await queue.follow("r1")

    # A worker fails the node and reports it; no coordinator ever reads the report.
    await queue.submit(WorkItem("r1", "bad", 0))
    await Worker(name="w0", registry=registry, store=store, queue=queue).step()

    record, _ = await run_across(
        None,
        registry=registry,
        store=store,
        queue=queue,
        workers=1,
        policy=RunPolicy(failure_mode=FailureMode.SKIP_DOWNSTREAM),
        resume=True,
    )

    assert record.state is RunState.FAILED
    assert record.nodes["bad"].state is NodeState.FAILED
    assert record.nodes["blocked"].state is NodeState.SKIPPED


async def test_a_node_resumed_at_a_later_attempt_keeps_that_attempt_across_the_queue() -> None:
    # The crash test above cannot distinguish "uses the attempt it was handed" from "always
    # starts at zero", because every attempt in it *is* zero. This one puts a node back at
    # attempt 2, which is the only way to see that the number really travels in the message
    # and reaches the agent — and therefore that the key an agent deduplicates on is the one
    # the outside world already saw.
    store, queue = InMemoryStateStore(), InMemoryWorkQueue()
    ledger: dict[str, NodeOutput] = {}
    effect = SideEffectAgent(ledger)
    registry = registry_with(effect=lambda: effect)
    workflow = WorkflowBuilder("solo").add_node("a", "effect").build()

    await store.save_workflow("r1", workflow)
    await store.checkpoint(
        RunStateRecord(
            run_id="r1",
            workflow_name="solo",
            state=RunState.RUNNING,
            nodes={
                "a": NodeStateRecord(run_id="r1", node_id="a", state=NodeState.RUNNING, attempt=2)
            },
        )
    )

    record, _ = await run_across(
        None, registry=registry, store=store, queue=queue, workers=1, resume=True
    )

    assert record.state is RunState.SUCCEEDED
    assert record.nodes["a"].attempt == 2
    assert list(ledger) == ["r1:a:2"]
    assert effect.commits == 1
