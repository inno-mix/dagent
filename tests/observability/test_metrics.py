"""Phase 7 acceptance: the six numbers FR-9 asks for, and that they are scrapeable.

Recorded through a real SDK meter provider into an in-memory reader, so these assert on
what a metrics backend would actually receive rather than on which methods were called.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from dagent.agents.fake import FailingAgent, FakeAgent, FlakyAgent
from dagent.graph.builder import WorkflowBuilder
from dagent.models.model_call import ModelRequest
from dagent.models.state import NodeOutput, NodeState
from dagent.models.workflow import Policy, Workflow
from dagent.observability import metrics as obs_metrics
from dagent.policy.limits import Budget
from dagent.policy.retry import Backoff, no_jitter
from dagent.policy.run import RunPolicy
from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import ManualClock
from dagent.runtime.executor import Executor
from dagent.runtime.model import StubModelClient
from dagent.runtime.registry import AgentRegistry
from dagent.store.memory import InMemoryStateStore


@pytest.fixture
def reader() -> Iterator[InMemoryMetricReader]:
    """A meter provider whose instruments dagent's executor will pick up.

    Injected rather than installed globally: `Metrics` takes a meter, so a test can hand
    the executor its own without touching process-wide state and without racing any other
    test that wanted a different one.
    """
    collected = InMemoryMetricReader()
    yield collected


def instruments(source: InMemoryMetricReader) -> obs_metrics.Metrics:
    return obs_metrics.Metrics(MeterProvider(metric_readers=[source]).get_meter("dagent"))


def executor_with(reader: InMemoryMetricReader, **kwargs: Any) -> Executor:
    executor = Executor(store=InMemoryStateStore(), **kwargs)
    executor._metrics = instruments(reader)
    return executor


def points(collected: InMemoryMetricReader, name: str) -> list[Any]:
    """Every data point recorded under one instrument name."""
    data = collected.get_metrics_data()
    if data is None:
        return []
    return [
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def registry_with(**agents: object) -> AgentRegistry:
    registry = AgentRegistry()
    for name, factory in agents.items():
        registry.add(name, factory)  # type: ignore[arg-type]
    return registry


def diamond() -> Workflow:
    return (
        WorkflowBuilder("diamond")
        .add_node("a", "fake")
        .add_node("b", "fake", inputs={"x": "a"})
        .add_node("c", "fake", inputs={"x": "a"})
        .add_node("d", "fake", inputs={"l": "b", "r": "c"})
        .build()
    )


# --- the six numbers ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_flight_is_recorded_and_returns_to_zero(reader: InMemoryMetricReader) -> None:
    # An UpDownCounter that does not come back down is a dashboard that says the engine
    # is permanently busy.
    await executor_with(reader, registry=registry_with(fake=FakeAgent)).run(diamond(), run_id="r1")

    assert sum(point.value for point in points(reader, "dagent.nodes.in_flight")) == 0


@pytest.mark.asyncio
async def test_per_provider_in_flight_is_attributed_to_the_provider(
    reader: InMemoryMetricReader,
) -> None:
    await executor_with(
        reader, registry=registry_with(fake=FakeAgent), model=StubModelClient()
    ).run(diamond(), run_id="r1")

    recorded = points(reader, "dagent.provider.in_flight")
    assert recorded
    assert {point.attributes[obs_metrics.PROVIDER] for point in recorded} == {"stub"}


@pytest.mark.asyncio
async def test_ready_set_size_is_recorded_on_every_pass(reader: InMemoryMetricReader) -> None:
    # The backpressure story: a diamond offers 1, then 2, then 1, then 0.
    await executor_with(reader, registry=registry_with(fake=FakeAgent)).run(diamond(), run_id="r1")

    recorded = points(reader, "dagent.ready_set.size")
    assert recorded
    assert recorded[0].count >= 4  # one record per loop pass
    assert recorded[0].max == 2  # the fan-out


@pytest.mark.asyncio
async def test_retries_are_counted_only_when_a_node_actually_retried(
    reader: InMemoryMetricReader,
) -> None:
    workflow = (
        WorkflowBuilder("flaky")
        .add_node("a", "flaky", policy=Policy(max_attempts=3, backoff_initial_s=1.0))
        .build()
    )

    await executor_with(
        reader,
        registry=registry_with(flaky=lambda: FlakyAgent(fail_until_attempt=2)),
        clock=ManualClock(),
        policy=RunPolicy(backoff=Backoff(jitter=no_jitter)),
    ).run(workflow, run_id="r1")

    # Three attempts, two of them retries.
    assert sum(point.value for point in points(reader, "dagent.node.retries")) == 2


@pytest.mark.asyncio
async def test_a_run_with_no_retries_counts_none(reader: InMemoryMetricReader) -> None:
    await executor_with(reader, registry=registry_with(fake=FakeAgent)).run(diamond(), run_id="r1")

    assert sum(point.value for point in points(reader, "dagent.node.retries")) == 0


@pytest.mark.asyncio
async def test_run_duration_is_recorded_against_the_final_state(
    reader: InMemoryMetricReader,
) -> None:
    await executor_with(reader, registry=registry_with(fake=FakeAgent)).run(diamond(), run_id="r1")

    recorded = points(reader, "dagent.run.duration")
    assert len(recorded) == 1
    assert recorded[0].attributes[obs_metrics.STATE] == "succeeded"
    assert recorded[0].attributes[obs_metrics.WORKFLOW] == "diamond"


@pytest.mark.asyncio
async def test_tokens_are_counted_against_the_provider(reader: InMemoryMetricReader) -> None:
    class Caller:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            return (await ctx.model.complete(ModelRequest(prompt="one two three"))).text

    await executor_with(
        reader,
        registry=registry_with(caller=Caller),
        model=StubModelClient(lambda request: "a b"),
    ).run(WorkflowBuilder("call").add_node("n", "caller").build(), run_id="r1")

    recorded = points(reader, "dagent.model.tokens")
    assert sum(point.value for point in recorded) == 5
    assert recorded[0].attributes[obs_metrics.PROVIDER] == "stub"


@pytest.mark.asyncio
async def test_cost_is_counted_when_the_run_prices_its_calls(
    reader: InMemoryMetricReader,
) -> None:
    class Caller:
        async def run(self, ctx: AgentContext) -> NodeOutput:
            return (await ctx.model.complete(ModelRequest(prompt="one two three"))).text

    await executor_with(
        reader,
        registry=registry_with(caller=Caller),
        model=StubModelClient(lambda request: "a b"),
        policy=RunPolicy(budget=Budget(), price=lambda response: 0.5),
    ).run(WorkflowBuilder("call").add_node("n", "caller").build(), run_id="r1")

    assert sum(point.value for point in points(reader, "dagent.model.cost")) == 0.5


@pytest.mark.asyncio
async def test_a_run_that_priced_nothing_records_no_cost(reader: InMemoryMetricReader) -> None:
    # The default Pricer is free, and a stream of zeros is worse than no series at all.
    await executor_with(reader, registry=registry_with(fake=FakeAgent)).run(diamond(), run_id="r1")

    assert points(reader, "dagent.model.cost") == []


@pytest.mark.asyncio
async def test_completions_are_counted_by_the_state_they_reached(
    reader: InMemoryMetricReader,
) -> None:
    workflow = WorkflowBuilder("mixed").add_node("ok", "fake").add_node("bad", "boom").build()

    await executor_with(reader, registry=registry_with(fake=FakeAgent, boom=FailingAgent)).run(
        workflow, run_id="r1"
    )

    by_state = {
        point.attributes[obs_metrics.STATE]: point.value
        for point in points(reader, "dagent.nodes.completed")
    }
    assert by_state == {NodeState.SUCCESS.value: 1, NodeState.FAILED.value: 1}


# --- attributes stay low cardinality -------------------------------------------------------


@pytest.mark.asyncio
async def test_no_metric_is_attributed_by_run_id_or_node_id(
    reader: InMemoryMetricReader,
) -> None:
    # One time series per run is how a metrics backend falls over. Per-run detail is what
    # the trace and the inspector are for.
    await executor_with(
        reader, registry=registry_with(fake=FakeAgent), model=StubModelClient()
    ).run(diamond(), run_id="r1")

    data = reader.get_metrics_data()
    assert data is not None
    keys = {
        key
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        for point in metric.data.data_points
        for key in (point.attributes or {})
    }
    assert "run_id" not in keys
    assert "node_id" not in keys
    assert keys <= {
        obs_metrics.PROVIDER,
        obs_metrics.AGENT,
        obs_metrics.WORKFLOW,
        obs_metrics.STATE,
    }


def test_the_default_meter_is_a_no_op_that_still_builds_every_instrument() -> None:
    # Without an SDK the API returns no-op instruments. The engine records into them on
    # every transition, so they have to exist and accept calls.
    built = obs_metrics.metrics_for()

    built.nodes_in_flight.add(1, {})
    built.ready_set_size.record(3, {})
    built.tokens.add(10, {})
