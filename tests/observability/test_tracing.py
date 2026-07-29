"""Phase 7 acceptance: the trace a run produces matches the run.

The SDK is a dev dependency, so these use a real tracer provider with an in-memory
exporter — the actual OpenTelemetry machinery, only the transport replaced. Anything less
would be testing that the code calls functions, not that it produces a trace.
"""

import asyncio
from collections.abc import Iterator, Sequence

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dagent.agents.fake import FailingAgent, FakeAgent, FlakyAgent
from dagent.graph.builder import WorkflowBuilder
from dagent.models.state import NodeOutput, RunState
from dagent.models.workflow import Policy
from dagent.observability import setup, tracing
from dagent.policy.retry import Backoff, no_jitter
from dagent.policy.run import RunPolicy
from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import ManualClock
from dagent.runtime.executor import Executor
from dagent.runtime.model import StubModelClient
from dagent.runtime.registry import AgentRegistry
from dagent.store.memory import InMemoryStateStore

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="session", autouse=True)
def tracer_provider() -> None:
    """Attach an in-memory exporter to whichever tracer provider this process has.

    `set_tracer_provider` is deliberately one-way — OpenTelemetry will not let a process
    change its mind — so installing our own would lose a race with any other test module
    that configured one first, and reaching past it into the module global is how you
    restore the proxy provider onto itself and recurse forever. Attaching a processor to
    the live provider works whoever installed it, and goes through `setup.configure`, so
    the production wiring is on the path rather than beside it.
    """
    setup.configure(traces=True, metrics=False, logs=False)
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider), "expected an SDK provider"
    provider.add_span_processor(SimpleSpanProcessor(EXPORTER))


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    """Hand each test an exporter holding only its own spans."""
    EXPORTER.clear()
    yield EXPORTER
    EXPORTER.clear()


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


def by_name(recorded: Sequence[ReadableSpan]) -> dict[str, ReadableSpan]:
    return {span.name: span for span in recorded}


async def run_diamond(**kwargs: object) -> object:
    return await Executor(
        registry=registry_with(fake=FakeAgent),
        store=InMemoryStateStore(),
        **kwargs,  # type: ignore[arg-type]
    ).run(diamond(), run_id="r1")  # type: ignore[arg-type]


# --- the acceptance criterion -----------------------------------------------------------


@pytest.mark.asyncio
async def test_every_node_produces_a_span_nested_under_the_run_span(
    spans: InMemorySpanExporter,
) -> None:
    await run_diamond()

    recorded = spans.get_finished_spans()
    names = {span.name for span in recorded}
    assert names == {
        "dagent.run diamond",
        "dagent.node a",
        "dagent.node b",
        "dagent.node c",
        "dagent.node d",
    }

    run = by_name(recorded)["dagent.run diamond"]
    for node_id in ("a", "b", "c", "d"):
        node = by_name(recorded)[f"dagent.node {node_id}"]
        assert node.parent is not None
        assert node.parent.span_id == run.context.span_id
        assert node.context.trace_id == run.context.trace_id


@pytest.mark.asyncio
async def test_the_span_tree_covers_exactly_the_nodes_the_run_executed(
    spans: InMemorySpanExporter,
) -> None:
    # "Matches the DAG": a node that ran has a span, a node that did not has none. Here
    # `blocked` never becomes ready, so a span for it would be a trace that lies.
    workflow = (
        WorkflowBuilder("partial")
        .add_node("bad", "boom")
        .add_node("blocked", "fake", inputs={"x": "bad"})
        .build()
    )

    await Executor(
        registry=registry_with(fake=FakeAgent, boom=FailingAgent), store=InMemoryStateStore()
    ).run(workflow, run_id="r1")

    names = {span.name for span in spans.get_finished_spans()}
    assert "dagent.node bad" in names
    assert "dagent.node blocked" not in names


@pytest.mark.asyncio
async def test_a_retried_node_gets_one_span_per_attempt(spans: InMemorySpanExporter) -> None:
    # Per attempt, not per node: three tries took three different times for three
    # different reasons, and one averaged span throws away what you opened the trace for.
    workflow = (
        WorkflowBuilder("flaky")
        .add_node("a", "flaky", policy=Policy(max_attempts=3, backoff_initial_s=1.0))
        .build()
    )

    await Executor(
        registry=registry_with(flaky=lambda: FlakyAgent(fail_until_attempt=2)),
        store=InMemoryStateStore(),
        clock=ManualClock(),
        policy=RunPolicy(backoff=Backoff(jitter=no_jitter)),
    ).run(workflow, run_id="r1")

    node_spans = [s for s in spans.get_finished_spans() if s.name == "dagent.node a"]
    assert len(node_spans) == 3
    assert sorted(s.attributes[tracing.ATTEMPT] for s in node_spans) == [0, 1, 2]  # type: ignore[index]


# --- attributes ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_run_span_carries_the_run_id_and_final_state(
    spans: InMemorySpanExporter,
) -> None:
    await run_diamond()

    run = by_name(spans.get_finished_spans())["dagent.run diamond"]
    assert run.attributes[tracing.RUN_ID] == "r1"  # type: ignore[index]
    assert run.attributes[tracing.WORKFLOW] == "diamond"  # type: ignore[index]
    assert run.attributes[tracing.RUN_STATE] == RunState.SUCCEEDED.value  # type: ignore[index]


@pytest.mark.asyncio
async def test_a_node_span_carries_attempt_agent_and_provider(
    spans: InMemorySpanExporter,
) -> None:
    # FR-9 names these three specifically.
    await run_diamond(model=StubModelClient())

    node = by_name(spans.get_finished_spans())["dagent.node b"]
    assert node.attributes[tracing.NODE_ID] == "b"  # type: ignore[index]
    assert node.attributes[tracing.AGENT] == "fake"  # type: ignore[index]
    assert node.attributes[tracing.ATTEMPT] == 0  # type: ignore[index]
    assert node.attributes[tracing.PROVIDER] == "stub"  # type: ignore[index]


@pytest.mark.asyncio
async def test_a_node_that_called_a_model_carries_its_token_usage(
    spans: InMemorySpanExporter,
) -> None:
    class Caller:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            from dagent.models.model_call import ModelRequest

            return (await ctx.model.complete(ModelRequest(prompt="one two three"))).text

    await Executor(
        registry=registry_with(caller=Caller),
        store=InMemoryStateStore(),
        model=StubModelClient(lambda request: "a b"),
    ).run(WorkflowBuilder("call").add_node("n", "caller").build(), run_id="r1")

    node = by_name(spans.get_finished_spans())["dagent.node n"]
    assert node.attributes[tracing.OUTPUT_TOKENS] == 5  # type: ignore[index]


@pytest.mark.asyncio
async def test_a_node_that_called_nothing_reports_no_token_attribute(
    spans: InMemorySpanExporter,
) -> None:
    # Zero and absent mean different things. A node that never called a model has no
    # token count, and recording a 0 would make it look like one that called and got
    # nothing back.
    await run_diamond()

    node = by_name(spans.get_finished_spans())["dagent.node a"]
    assert tracing.OUTPUT_TOKENS not in (node.attributes or {})


# --- the default costs nothing --------------------------------------------------------------


def test_the_tracer_is_resolved_per_call_not_captured_at_import(
    spans: InMemorySpanExporter,
) -> None:
    # An application configures its provider after importing dagent, which is the normal
    # order — as this very module does. A tracer captured at import time would have
    # missed it and every span here would be lost.
    with tracing.run_span("late", "configured"):
        pass

    assert [span.name for span in spans.get_finished_spans()] == ["dagent.run configured"]


@pytest.mark.asyncio
async def test_concurrent_nodes_share_one_trace(spans: InMemorySpanExporter) -> None:
    # Nesting comes from contextvars, which asyncio copies into each task. If that broke,
    # concurrent branches would each start their own trace and the tree would fall apart.
    await asyncio.wait_for(run_diamond(), timeout=5)

    traces = {span.context.trace_id for span in spans.get_finished_spans()}
    assert len(traces) == 1
