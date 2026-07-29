"""Turning the no-op tracer and meter into real ones.

The rest of this package emits through OpenTelemetry's API, which does nothing until a
provider is installed. This is the one module that installs one, and it is the only place
in dagent that imports the SDK — so a caller who never calls :func:`configure` never pays
for it, and never has to have the ``otel`` extra installed at all.

Deliberately an *application* concern. A library that configures global providers on import
takes a decision away from the process that owns it, and gets it wrong the moment two
libraries do it.
"""

from __future__ import annotations

import importlib.util

from dagent.errors import DagentError
from dagent.observability.logging import configure as configure_logging

__all__ = ["configure", "prometheus_text"]

EXTRA_HINT = "install the observability extra: pip install 'dagent[otel]'"


def configure(
    *,
    traces: bool = True,
    metrics: bool = True,
    logs: bool = True,
    log_level: int | str = "INFO",
    json_logs: bool = False,
    console_traces: bool = False,
) -> None:
    """Install real providers for whichever signals are wanted.

    Metrics go to a Prometheus reader, which registers with ``prometheus_client``'s
    default registry — so anything that already scrapes that process picks dagent's
    instruments up with no further wiring, and :func:`prometheus_text` renders them for a
    caller who would rather serve the text itself.

    Traces get a provider with no exporter unless ``console_traces`` is set. A provider
    with no exporter still builds the span tree, which is what an in-process consumer or a
    test needs; shipping spans somewhere is the application's business.

    Args:
        traces: Install a tracer provider.
        metrics: Install a meter provider with a Prometheus reader.
        logs: Configure ``structlog``.
        log_level: Passed to the logging configuration.
        json_logs: Render logs as JSON.
        console_traces: Also print finished spans to stdout. Useful once, in development.

    Raises:
        DagentError: If the SDK is not installed.
    """
    if logs:
        configure_logging(level=log_level, json=json_logs)

    if not (traces or metrics):
        return
    if importlib.util.find_spec("opentelemetry.sdk") is None:
        raise DagentError(f"OpenTelemetry SDK is not installed; {EXTRA_HINT}")

    if traces:
        _configure_traces(console=console_traces)
    if metrics:
        _configure_metrics()


def _configure_traces(*, console: bool) -> None:
    """Install a tracer provider, optionally printing spans as they finish."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "dagent"}))
    if console:
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


def _configure_metrics() -> None:
    """Install a meter provider reading into the Prometheus registry."""
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource

    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
    except ImportError as exc:  # pragma: no cover — only without the extra
        raise DagentError(f"the Prometheus exporter is not installed; {EXTRA_HINT}") from exc

    metrics.set_meter_provider(
        MeterProvider(
            resource=Resource.create({SERVICE_NAME: "dagent"}),
            metric_readers=[PrometheusMetricReader()],
        )
    )


def prometheus_text() -> str:
    """Render the current metrics in Prometheus' exposition format.

    What a ``/metrics`` endpoint would return. Offered as text rather than as a server
    because dagent does not own the process it runs in and should not decide to open a
    port in it — an application that wants one hands this to whatever it already serves.

    Raises:
        DagentError: If ``prometheus_client`` is not installed.
    """
    try:
        from prometheus_client import REGISTRY, generate_latest
    except ImportError as exc:  # pragma: no cover — only without the extra
        raise DagentError(f"prometheus_client is not installed; {EXTRA_HINT}") from exc

    return generate_latest(REGISTRY).decode("utf-8")
