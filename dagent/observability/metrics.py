"""The six numbers FR-9 asks for, as OpenTelemetry instruments.

Same API-only bargain as ``tracing``: without an SDK these record into nothing. With one,
they are scrapeable through whatever reader the application installed — the ``otel`` extra
wires a Prometheus one.

The choice of instrument per number is the part worth reading. In-flight counts are
``UpDownCounter`` because they go both ways and only the current value means anything;
ready-set size is a ``Histogram`` because its *distribution* is the backpressure story and
its instantaneous value is noise; retries and tokens are ``Counter`` because they only ever
accumulate and rate is what you want from them.
"""

from __future__ import annotations

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, Meter, UpDownCounter

__all__ = ["Metrics", "meter", "metrics_for"]

INSTRUMENTATION_NAME = "dagent"

PROVIDER = "provider"
AGENT = "agent"
WORKFLOW = "workflow"
STATE = "state"
"""Attribute keys. Deliberately low-cardinality: none of these is a run id or a node id.

A metric attributed by run id produces one time series per run, which is how a metrics
backend falls over. Per-run detail is what the trace and the inspector are for."""


def meter() -> Meter:
    """Return dagent's meter, resolved per call for the same reason the tracer is."""
    return metrics.get_meter(INSTRUMENTATION_NAME)


class Metrics:
    """Every instrument dagent records, built once per run.

    Instrument creation is not free — the SDK looks each one up and validates it — and the
    executor records on every node transition, so they are built once at the top of a run
    and passed down rather than resolved at each call site.
    """

    def __init__(self, source: Meter | None = None) -> None:
        """Create the instruments against ``source``, or against dagent's own meter."""
        made = source if source is not None else meter()

        self.nodes_in_flight: UpDownCounter = made.create_up_down_counter(
            "dagent.nodes.in_flight",
            unit="{node}",
            description="Nodes currently executing.",
        )
        self.provider_in_flight: UpDownCounter = made.create_up_down_counter(
            "dagent.provider.in_flight",
            unit="{node}",
            description="Nodes currently executing, by model provider.",
        )
        self.ready_set_size: Histogram = made.create_histogram(
            "dagent.ready_set.size",
            unit="{node}",
            description="Nodes the scheduler found ready on each pass of the run loop.",
        )
        self.retries: Counter = made.create_counter(
            "dagent.node.retries",
            unit="{attempt}",
            description="Node attempts made beyond the first.",
        )
        self.nodes_completed: Counter = made.create_counter(
            "dagent.nodes.completed",
            unit="{node}",
            description="Nodes that reached a terminal state, by that state.",
        )
        self.run_duration: Histogram = made.create_histogram(
            "dagent.run.duration",
            unit="s",
            description="Wall-clock seconds from a run opening to its terminal state.",
        )
        self.tokens: Counter = made.create_counter(
            "dagent.model.tokens",
            unit="{token}",
            description="Tokens charged to the run's budget.",
        )
        self.cost: Counter = made.create_counter(
            "dagent.model.cost",
            unit="USD",
            description="Cost charged to the run's budget, as priced by the run's Pricer.",
        )


def metrics_for(source: Meter | None = None) -> Metrics:
    """Build the instrument set for one run."""
    return Metrics(source)
