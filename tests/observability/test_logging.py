"""Structured logs, and the correlation FR-9 asks for.

The important property is the one that is easy to get wrong: a log line written inside an
agent, several frames and one `asyncio` task away from the scheduler, still knows which
run and which node caused it — without anybody having passed it down.
"""

import asyncio
import logging
from collections.abc import Iterator

import pytest
import structlog

from dagent.agents.fake import FailingAgent, FakeAgent
from dagent.graph.builder import WorkflowBuilder
from dagent.models.state import NodeOutput
from dagent.observability import logging as obs
from dagent.runtime.agent import AgentContext
from dagent.runtime.executor import Executor
from dagent.runtime.registry import AgentRegistry
from dagent.store.memory import InMemoryStateStore


@pytest.fixture
def captured() -> Iterator[list[dict[str, object]]]:
    """Capture structlog events as data, before any renderer turns them into text."""
    entries: list[dict[str, object]] = []

    def capture(_logger: object, _name: str, event: dict[str, object]) -> None:
        entries.append(dict(event))
        # Swallow it: the chain ends here, so nothing has to render and no logger has to
        # accept whatever a capturing processor would otherwise have returned.
        raise structlog.DropEvent

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            capture,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    try:
        yield entries
    finally:
        structlog.reset_defaults()


def registry_with(**agents: object) -> AgentRegistry:
    registry = AgentRegistry()
    for name, factory in agents.items():
        registry.add(name, factory)  # type: ignore[arg-type]
    return registry


# --- correlation ---------------------------------------------------------------------------


def test_bind_run_tags_lines_with_the_run(captured: list[dict[str, object]]) -> None:
    with obs.bind_run("r1", "wf"):
        obs.get_logger().info("something")

    assert captured[0]["run_id"] == "r1"
    assert captured[0]["workflow"] == "wf"


def test_bindings_are_released_on_the_way_out(captured: list[dict[str, object]]) -> None:
    # Left bound, a node's identity would leak onto whatever the scheduler did next.
    with obs.bind_run("r1", "wf"):
        pass
    obs.get_logger().info("after")

    assert "run_id" not in captured[0]


def test_node_bindings_nest_inside_run_bindings(captured: list[dict[str, object]]) -> None:
    with obs.bind_run("r1", "wf"), obs.bind_node("n", "fake", 2):
        obs.get_logger().info("inside")

    entry = captured[0]
    assert (entry["run_id"], entry["node_id"], entry["agent"], entry["attempt"]) == (
        "r1",
        "n",
        "fake",
        2,
    )


@pytest.mark.asyncio
async def test_a_line_written_inside_an_agent_carries_its_node(
    captured: list[dict[str, object]],
) -> None:
    # The whole point. The agent is handed no logger and no ids; contextvars carry them
    # across the task boundary the executor created.
    class Chatty:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            obs.get_logger().info("agent.spoke")
            return None

    await Executor(registry=registry_with(chatty=Chatty), store=InMemoryStateStore()).run(
        WorkflowBuilder("wf").add_node("n", "chatty").build(), run_id="r1"
    )

    spoke = [entry for entry in captured if entry.get("event") == "agent.spoke"]
    assert len(spoke) == 1
    assert (spoke[0]["run_id"], spoke[0]["node_id"]) == ("r1", "n")


@pytest.mark.asyncio
async def test_concurrent_nodes_do_not_borrow_each_others_identity(
    captured: list[dict[str, object]],
) -> None:
    # Two nodes in flight at once, each logging after a suspension point. If the binding
    # were process-global rather than per-task context, they would cross.
    barrier = asyncio.Barrier(2)

    class Overlapping:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            await barrier.wait()
            obs.get_logger().info("agent.spoke", saw=ctx.node_id)
            return None

    workflow = WorkflowBuilder("wf").add_node("left", "probe").add_node("right", "probe").build()

    await asyncio.wait_for(
        Executor(registry=registry_with(probe=Overlapping), store=InMemoryStateStore()).run(
            workflow, run_id="r1"
        ),
        timeout=5,
    )

    spoke = [entry for entry in captured if entry.get("event") == "agent.spoke"]
    assert len(spoke) == 2
    for entry in spoke:
        assert entry["node_id"] == entry["saw"]


@pytest.mark.asyncio
async def test_the_executor_narrates_the_run(captured: list[dict[str, object]]) -> None:
    await Executor(registry=registry_with(fake=FakeAgent), store=InMemoryStateStore()).run(
        WorkflowBuilder("wf").add_node("n", "fake").build(), run_id="r1"
    )

    events = [entry.get("event") for entry in captured]
    assert "run.started" in events
    assert "node.succeeded" in events
    assert "run.finished" in events


@pytest.mark.asyncio
async def test_every_node_level_line_carries_the_node_that_caused_it(
    captured: list[dict[str, object]],
) -> None:
    # Asserting the event names alone was not enough: `node.succeeded` was being emitted
    # from outside the node binding and arrived with no `node_id` at all, which is the one
    # field FR-9 asks a node-level line to carry.
    workflow = WorkflowBuilder("wf").add_node("ok", "fake").add_node("bad", "boom").build()

    await Executor(
        registry=registry_with(fake=FakeAgent, boom=FailingAgent), store=InMemoryStateStore()
    ).run(workflow, run_id="r1")

    node_lines = [entry for entry in captured if str(entry.get("event", "")).startswith("node.")]
    assert node_lines
    for entry in node_lines:
        assert entry.get("run_id") == "r1", entry
        assert entry.get("node_id"), entry


# --- configuration ----------------------------------------------------------------------------


def test_configure_is_safe_to_call_twice() -> None:
    obs.configure(level="DEBUG")
    obs.configure(level="INFO", json=True)

    assert obs.get_logger() is not None
    structlog.reset_defaults()
    logging.getLogger().handlers.clear()


def test_a_library_that_was_never_configured_writes_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # structlog unconfigured prints to stdout at every level, which for a library means an
    # import that starts spraying into somebody's terminal. Routing through stdlib logging
    # is what keeps dagent silent until asked.
    structlog.reset_defaults()
    logging.getLogger("dagent").handlers.clear()

    obs.get_logger().info("should not appear")

    assert capsys.readouterr().out == ""
