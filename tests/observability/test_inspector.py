"""Phase 7 acceptance: `inspect` output fully reconstructs what happened in a run.

"Fully" is the load-bearing word, so these tests take it literally: every node the run
touched, the state it ended in, when it ran, how many attempts it took, what it produced,
every model call, and the definition as actually executed. Run against both stores through
the shared `store` fixture, because a report that only works in memory is not a report.
"""

import json

import pytest

from dagent.agents.fake import FailingAgent, FakeAgent, FlakyAgent
from dagent.errors import StoreError
from dagent.graph.builder import WorkflowBuilder, build_node
from dagent.models.model_call import ModelRequest
from dagent.models.state import NodeOutput
from dagent.models.workflow import Policy
from dagent.observability.inspector import inspect_run
from dagent.policy.retry import Backoff, no_jitter
from dagent.policy.run import RunPolicy
from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import ManualClock
from dagent.runtime.executor import Executor
from dagent.runtime.model import StubModelClient
from dagent.runtime.registry import AgentRegistry
from dagent.store.base import StateStore


def registry_with(**agents: object) -> AgentRegistry:
    registry = AgentRegistry()
    for name, factory in agents.items():
        registry.add(name, factory)  # type: ignore[arg-type]
    return registry


def diamond() -> object:
    return (
        WorkflowBuilder("diamond")
        .add_node("a", "fake")
        .add_node("b", "fake", inputs={"x": "a"})
        .add_node("c", "fake", inputs={"x": "a"})
        .add_node("d", "fake", inputs={"l": "b", "r": "c"})
        .build()
    )


async def ran_diamond(store: StateStore, run_id: str = "r1") -> dict[str, object]:
    await Executor(registry=registry_with(fake=FakeAgent), store=store).run(
        diamond(),
        run_id=run_id,  # type: ignore[arg-type]
    )
    return await inspect_run(store, run_id)


# --- the acceptance criterion ---------------------------------------------------------


@pytest.mark.asyncio
async def test_the_report_covers_every_node_the_run_touched(store: StateStore) -> None:
    report = await ran_diamond(store)

    assert [node["node_id"] for node in report["nodes"]] == ["a", "b", "c", "d"]  # type: ignore[index,union-attr]


@pytest.mark.asyncio
async def test_every_node_reports_its_state_timings_and_attempt(store: StateStore) -> None:
    report = await ran_diamond(store)

    for node in report["nodes"]:  # type: ignore[union-attr]
        assert node["state"] == "success"
        assert node["attempt"] == 0
        assert node["started_at"] is not None
        assert node["finished_at"] is not None
        assert node["duration_s"] is not None


@pytest.mark.asyncio
async def test_every_successful_node_reports_what_it_produced(store: StateStore) -> None:
    report = await ran_diamond(store)

    outputs = {node["node_id"]: node["output"] for node in report["nodes"]}  # type: ignore[index,union-attr]
    assert outputs["a"] == {"node_id": "a", "attempt": 0, "inputs": {}}
    assert outputs["d"]["inputs"].keys() == {"l", "r"}  # type: ignore[index]


@pytest.mark.asyncio
async def test_the_run_summary_answers_what_happened_in_one_line(store: StateStore) -> None:
    report = await ran_diamond(store)

    summary = report["run"]
    assert summary["run_id"] == "r1"  # type: ignore[index]
    assert summary["state"] == "succeeded"  # type: ignore[index]
    assert summary["node_count"] == 4  # type: ignore[index]
    assert summary["states"] == {"success": 4}  # type: ignore[index]
    assert summary["duration_s"] is not None  # type: ignore[index]


@pytest.mark.asyncio
async def test_the_report_is_json_serializable(store: StateStore) -> None:
    # It is printed as JSON by the CLI; a report that cannot be serialized is not one.
    report = await ran_diamond(store)

    assert json.loads(json.dumps(report))["run"]["run_id"] == "r1"


# --- the harder cases ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failure_reports_its_error_and_leaves_its_dependents_visible(
    store: StateStore,
) -> None:
    workflow = (
        WorkflowBuilder("failing")
        .add_node("bad", "boom")
        .add_node("blocked", "fake", inputs={"x": "bad"})
        .build()
    )
    await Executor(registry=registry_with(fake=FakeAgent, boom=FailingAgent), store=store).run(
        workflow, run_id="r1"
    )

    report = await inspect_run(store, "r1")

    nodes = {node["node_id"]: node for node in report["nodes"]}  # type: ignore[index,union-attr]
    assert nodes["bad"]["state"] == "failed"
    assert "RuntimeError" in nodes["bad"]["error"]
    assert "output" not in nodes["bad"]  # it produced none, and says so by omission
    assert nodes["blocked"]["state"] == "pending"
    assert report["run"]["states"] == {"failed": 1, "pending": 1}  # type: ignore[index]


@pytest.mark.asyncio
async def test_a_retried_node_reports_the_attempt_it_finally_took(store: StateStore) -> None:
    workflow = (
        WorkflowBuilder("flaky")
        .add_node("a", "flaky", policy=Policy(max_attempts=3, backoff_initial_s=1.0))
        .build()
    )
    await Executor(
        registry=registry_with(flaky=lambda: FlakyAgent(fail_until_attempt=2)),
        store=store,
        clock=ManualClock(),
        policy=RunPolicy(backoff=Backoff(jitter=no_jitter)),
    ).run(workflow, run_id="r1")

    report = await inspect_run(store, "r1")

    assert report["nodes"][0]["attempt"] == 2  # type: ignore[index]


@pytest.mark.asyncio
async def test_every_model_call_is_reported_with_its_key_and_usage(
    store: StateStore,
) -> None:
    class Caller:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            return (await ctx.model.complete(ModelRequest(prompt=f"ask {ctx.node_id}"))).text

    await Executor(
        registry=registry_with(caller=Caller),
        store=store,
        model=StubModelClient(lambda request: "one two"),
    ).run(WorkflowBuilder("calls").add_node("n", "caller").build(), run_id="r1")

    report = await inspect_run(store, "r1")

    call = report["model_calls"][0]  # type: ignore[index]
    assert (call["node_id"], call["attempt"], call["sequence"]) == ("n", 0, 0)
    assert call["provider"] == "stub"
    assert call["total_tokens"] == 4
    assert call["prompt"] == "ask n"


@pytest.mark.asyncio
async def test_the_definition_reported_is_the_one_that_actually_ran(
    store: StateStore,
) -> None:
    # A dynamically grown run is exactly the one whose shape nobody can look up in a file,
    # so the inspector reporting the *expanded* graph is the whole point of reporting it.
    class Planner:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            ctx.expand(build_node("grown", "fake", depends_on=[ctx.node_id]))
            return None

    await Executor(registry=registry_with(planner=Planner, fake=FakeAgent), store=store).run(
        WorkflowBuilder("dyn").add_node("plan", "planner").build(), run_id="r1"
    )

    report = await inspect_run(store, "r1")

    assert [node["id"] for node in report["workflow"]["nodes"]] == ["plan", "grown"]  # type: ignore[index]


@pytest.mark.asyncio
async def test_outputs_can_be_left_out_when_they_are_large(store: StateStore) -> None:
    report = await ran_diamond(store)
    lean = await inspect_run(store, "r1", outputs=False)

    assert all("output" in node for node in report["nodes"])  # type: ignore[union-attr]
    assert all("output" not in node for node in lean["nodes"])  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_inspecting_a_run_that_does_not_exist_says_so(store: StateStore) -> None:
    with pytest.raises(StoreError, match="unknown run"):
        await inspect_run(store, "never-happened")


@pytest.mark.asyncio
async def test_a_run_whose_definition_was_never_stored_still_reports(
    store: StateStore,
) -> None:
    # Only reachable for a record written by something other than the executor, but the
    # inspector's job is to report what is there rather than to insist on completeness.
    from dagent.models.state import RunStateRecord

    await store.checkpoint(RunStateRecord(run_id="bare", workflow_name="unknown"))

    report = await inspect_run(store, "bare")

    assert report["workflow"] is None
    assert report["nodes"] == []
