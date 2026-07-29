"""Phase 5 acceptance: kill it mid-run, reload, resume, get the same result.

The crash is a `BaseException` raised from inside the store — the one seam every node
transition passes through — so it escapes the run loop the way a dying process would,
rather than being caught and recorded as an ordinary node failure.

Every test here runs against both store implementations via the `store` fixture. The
Postgres parametrisation skips itself unless `DAGENT_TEST_POSTGRES_DSN` is set.
"""

import asyncio
import json

import pytest

from dagent.agents.fake import FailingAgent, FakeAgent, FlakyAgent, SideEffectAgent
from dagent.errors import AgentError, StoreError, ValidationError
from dagent.graph.builder import WorkflowBuilder
from dagent.models.model_call import ModelRequest
from dagent.models.state import NodeOutput, NodeState, RunState
from dagent.models.workflow import Policy, Workflow
from dagent.policy.limits import Budget
from dagent.policy.retry import Backoff, no_jitter
from dagent.policy.run import RunPolicy
from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import ManualClock
from dagent.runtime.executor import Executor
from dagent.runtime.model import StubModelClient
from dagent.runtime.registry import AgentRegistry
from dagent.store.base import StateStore


class Crash(BaseException):
    """Stands in for the process dying. Not an `Exception`, so nothing catches it."""


def registry_with(**agents: object) -> AgentRegistry:
    registry = AgentRegistry()
    for name, factory in agents.items():
        registry.add(name, factory)  # type: ignore[arg-type]
    return registry


class CrashingStore:
    """Wraps a store and dies once a chosen node reaches a chosen state.

    Delegation rather than inheritance so it works for every implementation the `store`
    fixture hands over — the crash has to be reproducible against Postgres too, or half
    the acceptance criterion is untested.
    """

    def __init__(self, inner: StateStore, *, crash_on: tuple[str, NodeState] | None = None):
        self.inner = inner
        self._crash_on = crash_on
        self.armed = crash_on is not None

    async def save_node_state(self, record):  # type: ignore[no-untyped-def]
        if self.armed and self._crash_on == (record.node_id, record.state):
            self.armed = False
            # Die *before* the write lands: the harshest moment, because the store's idea
            # of this node is now one transition out of date.
            raise Crash(f"process died writing {record.node_id}={record.state}")
        await self.inner.save_node_state(record)

    async def checkpoint(self, run):  # type: ignore[no-untyped-def]
        await self.inner.checkpoint(run)

    async def save_workflow(self, run_id: str, workflow: Workflow) -> None:
        await self.inner.save_workflow(run_id, workflow)

    async def append_output(self, run_id: str, node_id: str, output: NodeOutput) -> None:
        await self.inner.append_output(run_id, node_id, output)

    async def load_run(self, run_id: str):  # type: ignore[no-untyped-def]
        return await self.inner.load_run(run_id)

    async def load_workflow(self, run_id: str) -> Workflow:
        return await self.inner.load_workflow(run_id)

    async def load_output(self, run_id: str, node_id: str) -> NodeOutput:
        return await self.inner.load_output(run_id, node_id)

    async def append_model_call(self, record):  # type: ignore[no-untyped-def]
        await self.inner.append_model_call(record)

    async def load_model_calls(self, run_id: str):  # type: ignore[no-untyped-def]
        return await self.inner.load_model_calls(run_id)


def chain() -> Workflow:
    """a -> b -> c, so the crash lands squarely between two nodes."""
    return (
        WorkflowBuilder("chain")
        .add_node("a", "effect")
        .add_node("b", "effect", inputs={"x": "a"})
        .add_node("c", "effect", inputs={"x": "b"})
        .build()
    )


def executor_for(store: object, ledger: dict[str, NodeOutput], **kwargs: object) -> Executor:
    agent = SideEffectAgent(ledger)
    return Executor(
        registry=registry_with(effect=lambda: agent),
        store=store,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_the_crashing_store_satisfies_the_protocol(store: StateStore) -> None:
    assert isinstance(CrashingStore(store), StateStore)


# --- the acceptance criterion ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_crashed_run_resumes_to_the_same_result_without_repeating_a_side_effect(
    store: StateStore,
) -> None:
    # (a) no node's side effect ran twice, and (b) the final outputs are identical to an
    # uninterrupted run. Both halves, one test, because either alone is worth little.
    ledger: dict[str, NodeOutput] = {}
    crashing = CrashingStore(store, crash_on=("b", NodeState.SUCCESS))

    with pytest.raises(Crash):
        await executor_for(crashing, ledger).run(chain(), run_id="r1")

    interrupted = await store.load_run("r1")
    assert interrupted.nodes["a"].state is NodeState.SUCCESS
    assert interrupted.nodes["b"].state is NodeState.RUNNING  # died before SUCCESS landed
    assert interrupted.nodes["c"].state is NodeState.PENDING

    # A brand new executor over the same store: nothing carried over in memory.
    resumed = await executor_for(store, ledger).resume("r1")

    assert resumed.state is RunState.SUCCEEDED
    assert {record.state for record in resumed.nodes.values()} == {NodeState.SUCCESS}
    # (a) `b` re-ran, and committed nothing new: three nodes, three commits.
    assert len(ledger) == 3
    assert sorted(ledger) == ["r1:a:0", "r1:b:0", "r1:c:0"]


@pytest.mark.asyncio
async def test_the_resumed_outputs_are_byte_identical_to_an_uninterrupted_run(
    store: StateStore,
) -> None:
    # `fake` rather than `effect` here on purpose: its output contains no run id, so the
    # two runs are comparable byte for byte rather than "the same apart from the parts
    # that identify them". That is the acceptance criterion's word, and it deserves the
    # strict reading.
    registry = registry_with(fake=FakeAgent)
    workflow = (
        WorkflowBuilder("chain")
        .add_node("a", "fake")
        .add_node("b", "fake", inputs={"x": "a"})
        .add_node("c", "fake", inputs={"x": "b"})
        .build()
    )

    await Executor(registry=registry, store=store).run(workflow, run_id="clean")

    crashing = CrashingStore(store, crash_on=("b", NodeState.SUCCESS))
    with pytest.raises(Crash):
        await Executor(registry=registry, store=crashing).run(workflow, run_id="crashed")
    await Executor(registry=registry, store=store).resume("crashed")

    for node_id in ("a", "b", "c"):
        baseline = await store.load_output("clean", node_id)
        resumed = await store.load_output("crashed", node_id)
        assert json.dumps(resumed, sort_keys=True) == json.dumps(baseline, sort_keys=True)


@pytest.mark.asyncio
async def test_a_node_that_already_succeeded_is_not_run_again(store: StateStore) -> None:
    ledger: dict[str, NodeOutput] = {}
    crashing = CrashingStore(store, crash_on=("c", NodeState.RUNNING))

    with pytest.raises(Crash):
        await executor_for(crashing, ledger).run(chain(), run_id="r1")

    # `a` and `b` finished before the crash. Their agent must not be constructed again.
    started: list[str] = []

    class Counting:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            started.append(ctx.node_id)
            return {"node_id": ctx.node_id, "key": ctx.idempotency_key}

    await Executor(registry=registry_with(effect=Counting), store=store).resume("r1")

    assert started == ["c"]


@pytest.mark.asyncio
async def test_an_interrupted_node_resumes_under_the_same_idempotency_key(
    store: StateStore,
) -> None:
    # The whole design in one assertion. A new attempt number would hand the outside world
    # a key it has never seen, and the dedupe that makes re-execution safe would not fire.
    #
    # The node is crashed on its *third* attempt on purpose. Crash it on the first and
    # "resume at the attempt it reached" and "resume at zero" produce identical keys, so
    # the test would pass against a broken implementation — which is exactly what an
    # earlier version of this test did.
    keys: list[str] = []

    class KeyWatcher:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            keys.append(ctx.idempotency_key)
            if ctx.attempt < 2:
                raise AgentError("not yet")
            return None

    workflow = (
        WorkflowBuilder("single")
        .add_node("a", "effect", policy=Policy(max_attempts=6, backoff_initial_s=1.0))
        .build()
    )
    policy = RunPolicy(backoff=Backoff(jitter=no_jitter))
    crashing = CrashingStore(store, crash_on=("a", NodeState.SUCCESS))

    with pytest.raises(Crash):
        await Executor(
            registry=registry_with(effect=KeyWatcher),
            store=crashing,
            clock=ManualClock(),
            policy=policy,
        ).run(workflow, run_id="r1")

    assert keys == ["r1:a:0", "r1:a:1", "r1:a:2"]

    await Executor(
        registry=registry_with(effect=KeyWatcher),
        store=store,
        clock=ManualClock(),
        policy=RunPolicy(backoff=Backoff(jitter=no_jitter)),
    ).resume("r1")

    # The fourth execution reuses the third's key. It does not start over at zero, and it
    # does not invent a fresh attempt either — this work never provably failed.
    assert keys == ["r1:a:0", "r1:a:1", "r1:a:2", "r1:a:2"]


@pytest.mark.asyncio
async def test_an_interrupted_retry_resumes_at_the_attempt_it_reached(
    store: StateStore,
) -> None:
    # A node two attempts deep when the process died comes back as attempt 2, not 0 —
    # otherwise a crash would silently hand a node its retries all over again.
    crashing = CrashingStore(store, crash_on=("a", NodeState.SUCCESS))
    workflow = (
        WorkflowBuilder("flaky")
        .add_node("a", "flaky", policy=Policy(max_attempts=4, backoff_initial_s=1.0))
        .build()
    )
    policy = RunPolicy(backoff=Backoff(jitter=no_jitter))

    with pytest.raises(Crash):
        await Executor(
            registry=registry_with(flaky=lambda: FlakyAgent(fail_until_attempt=2)),
            store=crashing,
            clock=ManualClock(),
            policy=policy,
        ).run(workflow, run_id="r1")

    assert (await store.load_run("r1")).nodes["a"].attempt == 2

    resumed = await Executor(
        registry=registry_with(flaky=lambda: FlakyAgent(fail_until_attempt=2)),
        store=store,
        clock=ManualClock(),
        policy=RunPolicy(backoff=Backoff(jitter=no_jitter)),
    ).resume("r1")

    assert resumed.state is RunState.SUCCEEDED
    assert await store.load_output("r1", "a") == {"node_id": "a", "attempt": 2}


# --- what resume refuses, and what it reloads -----------------------------------------


@pytest.mark.asyncio
async def test_resuming_an_unknown_run_says_so(store: StateStore) -> None:
    with pytest.raises(StoreError, match="unknown run"):
        await Executor(registry=registry_with(fake=FakeAgent), store=store).resume("ghost")


@pytest.mark.asyncio
async def test_resuming_a_finished_run_is_refused(store: StateStore) -> None:
    executor = Executor(registry=registry_with(fake=FakeAgent), store=store)
    await executor.run(WorkflowBuilder("done").add_node("a", "fake").build(), run_id="r1")

    with pytest.raises(ValidationError, match="nothing to resume"):
        await executor.resume("r1")


@pytest.mark.asyncio
async def test_a_failed_node_is_not_revisited_by_resume(store: StateStore) -> None:
    # Resume continues what was interrupted. A node that reached a verdict keeps it.
    workflow = (
        WorkflowBuilder("failed")
        .add_node("bad", "boom")
        .add_node("after", "fake", inputs={"x": "bad"})
        .build()
    )
    executor = Executor(registry=registry_with(fake=FakeAgent, boom=FailingAgent), store=store)
    await executor.run(workflow, run_id="r1")

    resumed = await executor.resume("r1")

    assert resumed.state is RunState.FAILED
    assert resumed.nodes["bad"].state is NodeState.FAILED
    assert resumed.nodes["after"].state is NodeState.PENDING


@pytest.mark.asyncio
async def test_resume_runs_the_definition_that_was_stored_not_one_it_is_handed(
    store: StateStore,
) -> None:
    # `resume` takes no workflow at all, which is the point: a resumed run cannot drift
    # onto a different graph, and a Phase 6 graph grown at run time is still resumable.
    crashing = CrashingStore(store, crash_on=("b", NodeState.RUNNING))
    ledger: dict[str, NodeOutput] = {}

    with pytest.raises(Crash):
        await executor_for(crashing, ledger).run(chain(), run_id="r1")

    assert (await store.load_workflow("r1")).name == "chain"
    resumed = await executor_for(store, ledger).resume("r1")
    assert resumed.workflow_name == "chain"


@pytest.mark.asyncio
async def test_resume_refuses_a_definition_whose_agent_has_since_disappeared(
    store: StateStore,
) -> None:
    # Fail loud rather than half-run a graph the registry can no longer satisfy.
    ledger: dict[str, NodeOutput] = {}
    crashing = CrashingStore(store, crash_on=("b", NodeState.RUNNING))
    with pytest.raises(Crash):
        await executor_for(crashing, ledger).run(chain(), run_id="r1")

    with pytest.raises(ValidationError, match="not registered"):
        await Executor(registry=registry_with(other=FakeAgent), store=store).resume("r1")


@pytest.mark.asyncio
async def test_starting_a_second_run_under_an_existing_id_is_refused(
    store: StateStore,
) -> None:
    # Before persistence this was harmless; now it would clobber a resumable run.
    executor = Executor(registry=registry_with(fake=FakeAgent), store=store)
    workflow = WorkflowBuilder("one").add_node("a", "fake").build()
    await executor.run(workflow, run_id="r1")

    with pytest.raises(ValidationError, match="already exists"):
        await executor.run(workflow, run_id="r1")


# --- the budget survives the crash ----------------------------------------------------


class Caller:
    async def run(self, ctx: AgentContext) -> NodeOutput:
        response = await ctx.model.complete(ModelRequest(prompt=f"work for {ctx.node_id}"))
        return response.text


@pytest.mark.asyncio
async def test_a_resumed_run_remembers_what_it_already_spent(store: StateStore) -> None:
    # A per-run ceiling that reset itself every time the process died would not be a
    # ceiling — it would be a ceiling per crash.
    crashing = CrashingStore(store, crash_on=("b", NodeState.RUNNING))
    workflow = (
        WorkflowBuilder("spend")
        .add_node("a", "caller")
        .add_node("b", "caller", inputs={"x": "a"})
        .build()
    )

    with pytest.raises(Crash):
        await Executor(
            registry=registry_with(caller=Caller),
            store=crashing,
            model=StubModelClient(lambda request: "one two three"),
            policy=RunPolicy(budget=Budget(max_tokens=6)),
        ).run(workflow, run_id="r1")

    # `a` spent the whole ceiling before the crash. The resumed run must know that.
    budget = Budget(max_tokens=6)
    resumed = await Executor(
        registry=registry_with(caller=Caller),
        store=store,
        model=StubModelClient(lambda request: "one two three"),
        policy=RunPolicy(budget=budget),
    ).resume("r1")

    assert budget.tokens_used == 6
    assert resumed.state is RunState.BUDGET_EXCEEDED


@pytest.mark.asyncio
async def test_raising_the_ceiling_does_not_resurrect_a_node_that_already_failed(
    store: StateStore,
) -> None:
    # Worth pinning, because the opposite is the intuitive guess. Resume continues what
    # was *interrupted*; a node the budget refused reached a verdict, and re-running
    # verdicts is a different feature from crash recovery. Rehydration is still exact —
    # the carried-over spend is visible on the new budget — it just does not un-fail
    # anything.
    workflow = (
        WorkflowBuilder("spend")
        .add_node("a", "caller")
        .add_node("b", "caller", inputs={"x": "a"})
        .build()
    )
    await Executor(
        registry=registry_with(caller=Caller),
        store=store,
        model=StubModelClient(lambda request: "one two three"),
        policy=RunPolicy(budget=Budget(max_tokens=6)),
    ).run(workflow, run_id="r1")

    generous = Budget(max_tokens=1000)
    resumed = await Executor(
        registry=registry_with(caller=Caller),
        store=store,
        model=StubModelClient(lambda request: "one two three"),
        policy=RunPolicy(budget=generous),
    ).resume("r1")

    assert resumed.nodes["b"].state is NodeState.FAILED
    assert resumed.state is RunState.FAILED
    assert generous.tokens_used == 6  # carried over from before, and nothing new spent


@pytest.mark.asyncio
async def test_resume_is_a_no_op_when_there_is_nothing_left_to_do(
    store: StateStore,
) -> None:
    # Idempotent by construction: the loop finds no ready node and re-derives the state.
    workflow = WorkflowBuilder("fail").add_node("bad", "boom").build()
    executor = Executor(registry=registry_with(boom=FailingAgent), store=store)
    await executor.run(workflow, run_id="r1")

    first = await executor.resume("r1")
    second = await executor.resume("r1")

    assert first.state is second.state is RunState.FAILED
    assert first.nodes["bad"].error == second.nodes["bad"].error


@pytest.mark.asyncio
async def test_a_run_cancelled_from_outside_can_be_resumed(store: StateStore) -> None:
    entered = asyncio.Event()

    class Blocking:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            entered.set()
            await asyncio.Event().wait()
            return None

    workflow = WorkflowBuilder("blocking").add_node("a", "block").build()
    task = asyncio.create_task(
        Executor(registry=registry_with(block=Blocking), store=store).run(workflow, run_id="r1")
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await store.load_run("r1")).state is RunState.CANCELLED

    resumed = await Executor(registry=registry_with(block=FakeAgent), store=store).resume("r1")

    assert resumed.state is RunState.SUCCEEDED
