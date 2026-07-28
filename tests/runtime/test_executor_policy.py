"""Phase 4 acceptance: retries, timeouts, concurrency caps, budgets, failure semantics.

Every test here is offline and, apart from the two that must measure a real timeout,
instant: backoff sleeps go through a `ManualClock`, so a test can assert that a run waited
four seconds without waiting four seconds.
"""

import asyncio

import pytest

from dagent.agents.fake import FailingAgent, FakeAgent, FlakyAgent, HangingAgent
from dagent.errors import AgentError
from dagent.graph.builder import WorkflowBuilder
from dagent.models.model_call import ModelRequest
from dagent.models.state import NodeOutput, NodeState, RunState, RunStateRecord
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

DETERMINISTIC = Backoff(jitter=no_jitter)
"""Real backoff arithmetic, no randomness — so a test can assert on the delays."""


def registry_with(**agents: object) -> AgentRegistry:
    registry = AgentRegistry()
    for name, factory in agents.items():
        registry.add(name, factory)  # type: ignore[arg-type]
    return registry


class Counted:
    """Wraps an agent factory and counts how many times an attempt was actually made."""

    def __init__(self, build: object) -> None:
        self._build = build
        self.attempts = 0

    def __call__(self) -> object:
        self.attempts += 1
        return self._build()  # type: ignore[operator]


# --- acceptance 1: a flaky agent is retried to success --------------------------------


@pytest.mark.asyncio
async def test_an_agent_that_fails_twice_then_succeeds_is_retried_to_success() -> None:
    clock = ManualClock()
    store = InMemoryStateStore()
    factory = Counted(lambda: FlakyAgent(fail_until_attempt=2))
    workflow = (
        WorkflowBuilder("flaky")
        .add_node(
            "a",
            "flaky",
            policy=Policy(max_attempts=3, backoff_initial_s=2.0, backoff_max_s=100.0),
        )
        .build()
    )

    run = await Executor(
        registry=registry_with(flaky=factory),
        store=store,
        clock=clock,
        policy=RunPolicy(backoff=DETERMINISTIC),
    ).run(workflow, run_id="r1")

    assert run.state is RunState.SUCCEEDED
    assert run.nodes["a"].state is NodeState.SUCCESS
    assert factory.attempts == 3
    # The output proves the attempt number really reached the agent, rather than the
    # engine merely re-running something that happened to succeed the third time.
    assert await store.load_output("r1", "a") == {"node_id": "a", "attempt": 2}
    assert run.nodes["a"].attempt == 2


@pytest.mark.asyncio
async def test_retries_wait_the_backoff_the_policy_asked_for() -> None:
    clock = ManualClock()
    workflow = (
        WorkflowBuilder("flaky")
        .add_node(
            "a",
            "flaky",
            policy=Policy(max_attempts=3, backoff_initial_s=2.0, backoff_max_s=100.0),
        )
        .build()
    )

    await Executor(
        registry=registry_with(flaky=lambda: FlakyAgent(fail_until_attempt=2)),
        store=InMemoryStateStore(),
        clock=clock,
        policy=RunPolicy(backoff=DETERMINISTIC),
    ).run(workflow, run_id="r1")

    # Exponential, and through the injected clock — the run never really slept.
    assert clock.sleeps == [2.0, 4.0]


@pytest.mark.asyncio
async def test_a_node_without_its_own_policy_inherits_the_runs_defaults() -> None:
    clock = ManualClock()
    workflow = WorkflowBuilder("flaky").add_node("a", "flaky").build()

    run = await Executor(
        registry=registry_with(flaky=lambda: FlakyAgent(fail_until_attempt=1)),
        store=InMemoryStateStore(),
        clock=clock,
        policy=RunPolicy(node_defaults=Policy(max_attempts=2), backoff=DETERMINISTIC),
    ).run(workflow, run_id="r1")

    assert run.state is RunState.SUCCEEDED


@pytest.mark.asyncio
async def test_running_out_of_attempts_fails_the_node_with_the_last_error() -> None:
    clock = ManualClock()
    workflow = (
        WorkflowBuilder("flaky")
        .add_node("a", "flaky", policy=Policy(max_attempts=2, backoff_initial_s=1.0))
        .build()
    )

    run = await Executor(
        registry=registry_with(flaky=lambda: FlakyAgent(fail_until_attempt=99)),
        store=InMemoryStateStore(),
        clock=clock,
        policy=RunPolicy(backoff=DETERMINISTIC),
    ).run(workflow, run_id="r1")

    assert run.state is RunState.FAILED
    assert run.nodes["a"].state is NodeState.FAILED
    assert run.nodes["a"].attempt == 1
    assert "failed on attempt 1" in (run.nodes["a"].error or "")
    # One sleep, not two: nothing waits after the final attempt.
    assert clock.sleeps == [1.0]


@pytest.mark.asyncio
async def test_a_non_retryable_failure_is_not_retried_however_many_attempts_are_allowed() -> None:
    # FailingAgent raises a plain RuntimeError — a bug, not weather. Retrying it would
    # buy three identical stack traces.
    factory = Counted(FailingAgent)
    workflow = (
        WorkflowBuilder("permanent")
        .add_node("a", "boom", policy=Policy(max_attempts=5, backoff_initial_s=1.0))
        .build()
    )
    clock = ManualClock()

    run = await Executor(
        registry=registry_with(boom=factory),
        store=InMemoryStateStore(),
        clock=clock,
        policy=RunPolicy(backoff=DETERMINISTIC),
    ).run(workflow, run_id="r1")

    assert run.nodes["a"].state is NodeState.FAILED
    assert factory.attempts == 1
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_a_retried_node_is_never_recorded_as_terminally_failed_in_between() -> None:
    # FAILED is terminal. Writing it between attempts would leave a window in which a
    # resuming Phase 5 executor reads "this node is done, and it failed".
    class Watcher(InMemoryStateStore):
        def __init__(self) -> None:
            super().__init__()
            self.states: list[tuple[NodeState, int]] = []

        async def save_node_state(self, record):  # type: ignore[no-untyped-def]
            self.states.append((record.state, record.attempt))
            await super().save_node_state(record)

    store = Watcher()
    workflow = (
        WorkflowBuilder("flaky")
        .add_node("a", "flaky", policy=Policy(max_attempts=3, backoff_initial_s=1.0))
        .build()
    )

    await Executor(
        registry=registry_with(flaky=lambda: FlakyAgent(fail_until_attempt=2)),
        store=store,
        clock=ManualClock(),
        policy=RunPolicy(backoff=DETERMINISTIC),
    ).run(workflow, run_id="r1")

    assert store.states == [
        (NodeState.READY, 0),
        (NodeState.RUNNING, 0),
        (NodeState.RUNNING, 1),  # the rising attempt is the evidence of a retry
        (NodeState.RUNNING, 2),
        (NodeState.SUCCESS, 2),
    ]


class FlakyCaller:
    """Calls the model, then fails on the first attempt only."""

    async def run(self, ctx: AgentContext) -> NodeOutput:
        response = await ctx.model.complete(ModelRequest(prompt=f"try {ctx.attempt}"))
        if ctx.attempt == 0:
            raise AgentError("the call landed but the node failed afterwards")
        return response.text


@pytest.mark.asyncio
async def test_each_attempts_model_calls_are_recorded_under_that_attempt() -> None:
    # (run_id, node_id, attempt, sequence) is the key a replay client looks calls up by,
    # so the attempt component has to actually vary between attempts.
    store = InMemoryStateStore()
    workflow = (
        WorkflowBuilder("recorded")
        .add_node("a", "caller", policy=Policy(max_attempts=2, backoff_initial_s=1.0))
        .build()
    )

    await Executor(
        registry=registry_with(caller=FlakyCaller),
        store=store,
        clock=ManualClock(),
        model=StubModelClient(lambda request: f"reply to {request.prompt}"),
        policy=RunPolicy(backoff=DETERMINISTIC),
    ).run(workflow, run_id="r1")

    calls = await store.load_model_calls("r1")
    assert [(call.attempt, call.sequence) for call in calls] == [(0, 0), (1, 0)]
    assert [call.request.prompt for call in calls] == ["try 0", "try 1"]


# --- acceptance 2: a hanging agent is cancelled at its timeout ------------------------


@pytest.mark.asyncio
async def test_a_hanging_agent_is_cancelled_at_its_timeout() -> None:
    agent = HangingAgent()
    workflow = WorkflowBuilder("hang").add_node("a", "hang", policy=Policy(timeout_s=0.05)).build()

    run = await asyncio.wait_for(
        Executor(registry=registry_with(hang=lambda: agent), store=InMemoryStateStore()).run(
            workflow, run_id="r1"
        ),
        timeout=5,
    )

    assert run.state is RunState.FAILED
    assert run.nodes["a"].state is NodeState.FAILED
    assert "TimeoutError" in (run.nodes["a"].error or "")
    # Cancelled *cleanly*: the coroutine received its CancelledError at an await point and
    # got to run its own cleanup, rather than being abandoned mid-flight.
    assert (agent.started, agent.cancelled) == (1, 1)


@pytest.mark.asyncio
async def test_a_timed_out_node_is_retried_when_its_policy_allows() -> None:
    # A timeout is weather: the next attempt may well be fast enough.
    agent = HangingAgent()
    workflow = (
        WorkflowBuilder("hang")
        .add_node("a", "hang", policy=Policy(max_attempts=2, timeout_s=0.05, backoff_initial_s=1.0))
        .build()
    )

    run = await asyncio.wait_for(
        Executor(
            registry=registry_with(hang=lambda: agent),
            store=InMemoryStateStore(),
            clock=ManualClock(),
            policy=RunPolicy(backoff=DETERMINISTIC),
        ).run(workflow, run_id="r1"),
        timeout=5,
    )

    assert (agent.started, agent.cancelled) == (2, 2)
    assert run.nodes["a"].state is NodeState.FAILED


@pytest.mark.asyncio
async def test_a_node_that_finishes_inside_its_timeout_is_untouched() -> None:
    workflow = WorkflowBuilder("quick").add_node("a", "fake", policy=Policy(timeout_s=5)).build()

    run = await Executor(registry=registry_with(fake=FakeAgent), store=InMemoryStateStore()).run(
        workflow, run_id="r1"
    )

    assert run.state is RunState.SUCCEEDED


@pytest.mark.asyncio
async def test_a_timed_out_node_releases_its_permits() -> None:
    # Otherwise one slow node permanently shrinks the pool for everyone behind it.
    limits = Limits(max_concurrency=1)
    workflow = (
        WorkflowBuilder("hang-then-run")
        .add_node("slow", "hang", policy=Policy(timeout_s=0.05))
        .add_node("after", "fake")
        .build()
    )

    run = await asyncio.wait_for(
        Executor(
            registry=registry_with(hang=HangingAgent, fake=FakeAgent),
            store=InMemoryStateStore(),
            policy=RunPolicy(limits=limits),
        ).run(workflow, run_id="r1"),
        timeout=5,
    )

    assert run.nodes["slow"].state is NodeState.FAILED
    assert run.nodes["after"].state is NodeState.SUCCESS
    assert limits.in_flight("null") == 0


# --- acceptance 3: concurrency caps are never exceeded --------------------------------


class Overlapping:
    """Yields repeatedly so the scheduler has every chance to interleave, and counts."""

    def __init__(self, tally: dict[str, int]) -> None:
        self._tally = tally

    async def run(self, ctx: AgentContext) -> NodeOutput:
        self._tally["in_flight"] += 1
        self._tally["peak"] = max(self._tally["peak"], self._tally["in_flight"])
        for _ in range(5):
            await asyncio.sleep(0)
        self._tally["in_flight"] -= 1
        return ctx.node_id


def six_sources(agent_name: str = "probe") -> Workflow:
    builder = WorkflowBuilder("wide")
    for index in range(6):
        builder.add_node(f"n{index}", agent_name)
    return builder.build()


@pytest.mark.asyncio
async def test_with_no_cap_every_independent_node_runs_at_once() -> None:
    # The control for the next test: without a cap the executor really does dispatch all
    # six, so a cap of 2 holding the peak down is the cap working, not luck.
    tally = {"in_flight": 0, "peak": 0}

    await Executor(
        registry=registry_with(probe=lambda: Overlapping(tally)), store=InMemoryStateStore()
    ).run(six_sources(), run_id="r1")

    assert tally["peak"] == 6


@pytest.mark.asyncio
async def test_the_global_cap_holds_in_flight_nodes_down() -> None:
    tally = {"in_flight": 0, "peak": 0}
    limits = Limits(max_concurrency=2)

    run = await Executor(
        registry=registry_with(probe=lambda: Overlapping(tally)),
        store=InMemoryStateStore(),
        policy=RunPolicy(limits=limits),
    ).run(six_sources(), run_id="r1")

    assert tally["peak"] == 2
    assert run.state is RunState.SUCCEEDED


@pytest.mark.asyncio
async def test_the_per_provider_cap_holds_in_flight_nodes_for_that_provider_down() -> None:
    # FR-3: a node needs both permits, and the provider is the one its model client
    # reports — "stub" here, because that is the client this run shares.
    tally = {"in_flight": 0, "peak": 0}
    limits = Limits(per_provider={"stub": 2})

    run = await Executor(
        registry=registry_with(probe=lambda: Overlapping(tally)),
        store=InMemoryStateStore(),
        model=StubModelClient(),
        policy=RunPolicy(limits=limits),
    ).run(six_sources(), run_id="r1")

    assert tally["peak"] == 2
    assert limits.peak("stub") == 2
    assert run.state is RunState.SUCCEEDED


@pytest.mark.asyncio
async def test_the_ready_set_may_be_large_while_in_flight_stays_capped() -> None:
    # That gap is the backpressure. All six are ready at once; two run at a time.
    tally = {"in_flight": 0, "peak": 0}
    limits = Limits(max_concurrency=2)

    run = await Executor(
        registry=registry_with(probe=lambda: Overlapping(tally)),
        store=InMemoryStateStore(),
        policy=RunPolicy(limits=limits),
    ).run(six_sources(), run_id="r1")

    assert len(run.nodes) == 6
    assert {record.state for record in run.nodes.values()} == {NodeState.SUCCESS}
    assert limits.peak("null") == 2


# --- acceptance 4: the budget ends the run in BUDGET_EXCEEDED -------------------------


class Caller:
    """Makes exactly one model call and returns its text."""

    async def run(self, ctx: AgentContext) -> NodeOutput:
        response = await ctx.model.complete(ModelRequest(prompt=f"work for {ctx.node_id}"))
        return response.text


def chain_of_callers() -> Workflow:
    return (
        WorkflowBuilder("spend")
        .add_node("a", "caller")
        .add_node("b", "caller", inputs={"x": "a"})
        .add_node("c", "caller", inputs={"x": "b"})
        .build()
    )


@pytest.mark.asyncio
async def test_exceeding_the_budget_ends_the_run_in_budget_exceeded() -> None:
    client = StubModelClient(lambda request: "one two three")
    store = InMemoryStateStore()
    # "work for a" is three tokens in, "one two three" is three out: node a spends the
    # whole ceiling, and nothing after it is admitted.
    budget = Budget(max_tokens=6)

    run = await Executor(
        registry=registry_with(caller=Caller),
        store=store,
        model=client,
        policy=RunPolicy(budget=budget),
    ).run(chain_of_callers(), run_id="r1")

    assert run.state is RunState.BUDGET_EXCEEDED
    assert run.nodes["a"].state is NodeState.SUCCESS
    assert run.nodes["b"].state is NodeState.FAILED
    assert "budget exceeded" in (run.nodes["b"].error or "")


@pytest.mark.asyncio
async def test_no_further_model_calls_are_admitted_once_the_budget_is_spent() -> None:
    client = StubModelClient(lambda request: "one two three")

    await Executor(
        registry=registry_with(caller=Caller),
        store=InMemoryStateStore(),
        model=client,
        policy=RunPolicy(budget=Budget(max_tokens=6)),
    ).run(chain_of_callers(), run_id="r1")

    # The provider saw one request, not three. Admission is enforced at the seam every
    # call passes through rather than trusted to each agent.
    assert [request.prompt for request in client.requests] == ["work for a"]


@pytest.mark.asyncio
async def test_a_budget_refusal_is_not_retried() -> None:
    # Retrying an admission refusal is how you exceed a ceiling twice.
    client = StubModelClient(lambda request: "one two three")
    workflow = (
        WorkflowBuilder("spend")
        .add_node("a", "caller")
        .add_node("b", "caller", inputs={"x": "a"}, policy=Policy(max_attempts=4))
        .build()
    )
    clock = ManualClock()

    await Executor(
        registry=registry_with(caller=Caller),
        store=InMemoryStateStore(),
        clock=clock,
        model=client,
        policy=RunPolicy(budget=Budget(max_tokens=6), backoff=DETERMINISTIC),
    ).run(workflow, run_id="r1")

    assert len(client.requests) == 1
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_a_run_inside_its_budget_is_an_ordinary_success() -> None:
    client = StubModelClient(lambda request: "one two three")
    budget = Budget(max_tokens=10_000)

    run = await Executor(
        registry=registry_with(caller=Caller),
        store=InMemoryStateStore(),
        model=client,
        policy=RunPolicy(budget=budget),
    ).run(chain_of_callers(), run_id="r1")

    assert run.state is RunState.SUCCEEDED
    assert len(client.requests) == 3
    assert budget.tokens_used == 18


@pytest.mark.asyncio
async def test_a_ceiling_crossed_by_the_last_call_does_not_spoil_a_finished_run() -> None:
    # Exceeded but never refused: nothing was stopped, so this run succeeded.
    budget = Budget(max_tokens=1)
    workflow = WorkflowBuilder("one").add_node("a", "caller").build()

    run = await Executor(
        registry=registry_with(caller=Caller),
        store=InMemoryStateStore(),
        model=StubModelClient(lambda request: "one two three"),
        policy=RunPolicy(budget=budget),
    ).run(workflow, run_id="r1")

    assert run.state is RunState.SUCCEEDED
    assert (budget.exceeded, budget.refused) == (True, False)


@pytest.mark.asyncio
async def test_a_node_that_needs_no_model_is_unaffected_by_a_spent_budget() -> None:
    # The budget gates model calls, as FR-5 says, not dispatch. A node that costs nothing
    # is not punished for one that did.
    workflow = (
        WorkflowBuilder("mixed")
        .add_node("spender", "caller")
        .add_node("refused", "caller", inputs={"x": "spender"})
        .add_node("free", "fake")
        .build()
    )

    run = await Executor(
        registry=registry_with(caller=Caller, fake=FakeAgent),
        store=InMemoryStateStore(),
        model=StubModelClient(lambda request: "one two three"),
        policy=RunPolicy(budget=Budget(max_tokens=6)),
    ).run(workflow, run_id="r1")

    assert run.nodes["refused"].state is NodeState.FAILED
    assert run.nodes["free"].state is NodeState.SUCCESS
    assert run.state is RunState.BUDGET_EXCEEDED


# --- failure semantics ----------------------------------------------------------------


def branch_and_bystander() -> Workflow:
    """`bad` fails; `downstream` waits on it; `bystander` is independent."""
    return (
        WorkflowBuilder("semantics")
        .add_node("bad", "boom")
        .add_node("downstream", "fake", inputs={"x": "bad"})
        .add_node("bystander", "fake")
        .build()
    )


async def run_with(mode: FailureMode, workflow: Workflow | None = None) -> RunStateRecord:
    return await asyncio.wait_for(
        Executor(
            registry=registry_with(fake=FakeAgent, boom=FailingAgent, hang=HangingAgent),
            store=InMemoryStateStore(),
            policy=RunPolicy(failure_mode=mode),
        ).run(workflow if workflow is not None else branch_and_bystander(), run_id="r1"),
        timeout=5,
    )


@pytest.mark.asyncio
async def test_run_to_completion_lets_independent_branches_finish() -> None:
    run = await run_with(FailureMode.RUN_TO_COMPLETION)

    assert run.nodes["bad"].state is NodeState.FAILED
    assert run.nodes["bystander"].state is NodeState.SUCCESS
    # Left PENDING: this mode makes no claim about what will never happen.
    assert run.nodes["downstream"].state is NodeState.PENDING
    assert run.state is RunState.FAILED


@pytest.mark.asyncio
async def test_skip_downstream_marks_the_blast_radius_rather_than_leaving_it_pending() -> None:
    run = await run_with(FailureMode.SKIP_DOWNSTREAM)

    assert run.nodes["bad"].state is NodeState.FAILED
    assert run.nodes["downstream"].state is NodeState.SKIPPED
    assert "upstream node 'bad' failed" in (run.nodes["downstream"].error or "")
    # Still run-to-completion for everything not in the blast radius.
    assert run.nodes["bystander"].state is NodeState.SUCCESS
    assert run.state is RunState.FAILED


@pytest.mark.asyncio
async def test_skip_downstream_reaches_the_whole_transitive_blast_radius() -> None:
    workflow = (
        WorkflowBuilder("deep")
        .add_node("bad", "boom")
        .add_node("child", "fake", inputs={"x": "bad"})
        .add_node("grandchild", "fake", inputs={"x": "child"})
        .build()
    )

    run = await run_with(FailureMode.SKIP_DOWNSTREAM, workflow)

    assert run.nodes["child"].state is NodeState.SKIPPED
    assert run.nodes["grandchild"].state is NodeState.SKIPPED


@pytest.mark.asyncio
async def test_fail_fast_cancels_everything_still_running() -> None:
    workflow = (
        WorkflowBuilder("fail-fast")
        .add_node("slow", "hang")
        .add_node("bad", "boom")
        .add_node("later", "fake", inputs={"x": "slow"})
        .build()
    )

    run = await run_with(FailureMode.FAIL_FAST, workflow)

    assert run.nodes["bad"].state is NodeState.FAILED
    assert run.nodes["slow"].state is NodeState.FAILED
    assert "fail_fast after node 'bad' failed" in (run.nodes["slow"].error or "")
    # Nothing further was dispatched.
    assert run.nodes["later"].state is NodeState.PENDING
    assert run.state is RunState.FAILED


@pytest.mark.asyncio
async def test_fail_fast_leaves_no_node_task_behind() -> None:
    workflow = WorkflowBuilder("fail-fast").add_node("slow", "hang").add_node("bad", "boom").build()

    await run_with(FailureMode.FAIL_FAST, workflow)

    assert [task for task in asyncio.all_tasks() if task.get_name().startswith("dagent:")] == []


@pytest.mark.asyncio
async def test_a_node_that_finishes_despite_the_cancel_keeps_its_real_outcome() -> None:
    # An agent may catch CancelledError to clean up and still return. Recording that node
    # as cancelled would put a lie in the store; it completed.
    class Stubborn:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return {"finished": "anyway"}
            return None

    store = InMemoryStateStore()
    workflow = (
        WorkflowBuilder("stubborn")
        # Declared first, so it is dispatched and suspended before `bad` fails.
        .add_node("stubborn", "stubborn")
        .add_node("bad", "boom")
        .build()
    )

    run = await asyncio.wait_for(
        Executor(
            registry=registry_with(stubborn=Stubborn, boom=FailingAgent),
            store=store,
            policy=RunPolicy(failure_mode=FailureMode.FAIL_FAST),
        ).run(workflow, run_id="r1"),
        timeout=5,
    )

    assert run.nodes["stubborn"].state is NodeState.SUCCESS
    assert await store.load_output("r1", "stubborn") == {"finished": "anyway"}


@pytest.mark.asyncio
async def test_fail_fast_with_nothing_else_running_is_not_a_special_case() -> None:
    workflow = WorkflowBuilder("lonely").add_node("bad", "boom").build()

    run = await run_with(FailureMode.FAIL_FAST, workflow)

    assert run.state is RunState.FAILED


# --- external cancellation ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_cancelled_run_is_checkpointed_as_cancelled() -> None:
    # RunState.CANCELLED existed from Phase 1 and nothing had ever written it. A run that
    # was stopped from outside must not be left looking like it is still going.
    agent = HangingAgent()
    store = InMemoryStateStore()
    executor = Executor(registry=registry_with(hang=lambda: agent), store=store)
    workflow = WorkflowBuilder("hang").add_node("a", "hang").build()

    run_task = asyncio.create_task(executor.run(workflow, run_id="r1"))
    await asyncio.wait_for(agent.entered.wait(), timeout=5)
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    stored = await store.load_run("r1")
    assert stored.state is RunState.CANCELLED
    assert agent.cancelled == 1


@pytest.mark.asyncio
async def test_skip_downstream_marks_a_shared_descendant_only_once() -> None:
    # Two failures can share a blast radius. The second must not overwrite what the first
    # already recorded, or the node's stated cause would depend on completion order.
    workflow = (
        WorkflowBuilder("two-failures")
        .add_node("bad_one", "boom")
        .add_node("bad_two", "boom")
        .add_node("child", "fake", inputs={"x": "bad_one", "y": "bad_two"})
        .build()
    )

    run = await run_with(FailureMode.SKIP_DOWNSTREAM, workflow)

    assert run.nodes["child"].state is NodeState.SKIPPED
    assert "upstream node 'bad_one' failed" in (run.nodes["child"].error or "")


@pytest.mark.asyncio
async def test_a_keyboard_interrupt_still_stops_every_node_in_flight() -> None:
    # Ctrl-C is not an Exception, so it flies straight past the failure handling. It must
    # still not leave a node task running behind a process that is on its way out.
    class Interrupting(InMemoryStateStore):
        def __init__(self) -> None:
            super().__init__()
            self.writes = 0

        async def save_node_state(self, record):  # type: ignore[no-untyped-def]
            self.writes += 1
            if self.writes == 2:
                raise KeyboardInterrupt("ctrl-c")
            await super().save_node_state(record)

    workflow = WorkflowBuilder("interrupted").add_node("a", "hang").add_node("b", "hang").build()

    with pytest.raises(KeyboardInterrupt):
        await Executor(registry=registry_with(hang=HangingAgent), store=Interrupting()).run(
            workflow, run_id="r1"
        )

    assert [task for task in asyncio.all_tasks() if task.get_name().startswith("dagent:")] == []
