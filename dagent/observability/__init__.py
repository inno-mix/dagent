"""Traces, metrics, structured logs, and the run inspector (FR-9, FR-10).

Write-only, in both senses. Nothing here reads back into a run's outcome — a span that
fails to export cannot change what a node returned — and nothing here knows about a
specific agent. That is what lets the whole package be a no-op by default: dagent depends
on OpenTelemetry's *API*, which does nothing until an application installs an SDK, so an
engine that always emits costs a caller who wants nothing almost exactly nothing.

:mod:`dagent.observability.setup` is the one module that touches the SDK, and only when
called. Everything else works whether or not the ``otel`` extra is installed.
"""

from dagent.observability.inspector import inspect_run
from dagent.observability.logging import bind_node, bind_run, configure, get_logger
from dagent.observability.metrics import Metrics, metrics_for
from dagent.observability.tracing import node_span, run_span, tracer

__all__ = [
    "Metrics",
    "bind_node",
    "bind_run",
    "configure",
    "get_logger",
    "inspect_run",
    "metrics_for",
    "node_span",
    "run_span",
    "tracer",
]
