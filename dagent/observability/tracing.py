"""Spans: one per run, one per node attempt, nested (FR-9).

Only the OpenTelemetry **API** is imported here. Until an application installs an SDK the
tracer is a no-op whose overhead is a function call and an attribute lookup, so an engine
that always emits spans costs nothing to a caller who wants none. Installing the SDK — the
``otel`` extra, wired by :func:`dagent.observability.setup.configure` — is what turns the
same calls into a real trace.

Nesting falls out of ``contextvars``: ``asyncio.create_task`` copies the current context,
so a node task spawned inside the run span sees it as its parent without the executor
threading anything through. That is also why the span tree matches the DAG's *execution*
rather than its edges — a node's parent is the run, and the DAG's shape shows up in the
timing and the attributes rather than in the nesting.
"""

from __future__ import annotations

from contextlib import AbstractContextManager

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

__all__ = [
    "AGENT",
    "ATTEMPT",
    "ERROR",
    "INPUT_TOKENS",
    "NODE_ID",
    "NODE_STATE",
    "OUTPUT_TOKENS",
    "PROVIDER",
    "RUN_ID",
    "RUN_STATE",
    "WORKFLOW",
    "Span",
    "node_span",
    "run_span",
    "tracer",
]

INSTRUMENTATION_NAME = "dagent"

RUN_ID = "dagent.run_id"
NODE_ID = "dagent.node_id"
WORKFLOW = "dagent.workflow"
AGENT = "dagent.agent"
ATTEMPT = "dagent.attempt"
PROVIDER = "dagent.provider"
INPUT_TOKENS = "dagent.tokens.input"
OUTPUT_TOKENS = "dagent.tokens.output"
NODE_STATE = "dagent.node_state"
RUN_STATE = "dagent.run_state"
ERROR = "dagent.error"
"""Attribute names, defined once.

Strings scattered through the executor drift; a span attribute nobody can grep for is a
dashboard that quietly stops working. These are also what a test asserts against, so the
names are part of the contract rather than an implementation detail.
"""


def tracer() -> Tracer:
    """Return dagent's tracer.

    Fetched per call rather than cached at import: an application that configures its
    provider after importing dagent — which is the normal order — would otherwise be
    stuck with the no-op tracer for the life of the process.
    """
    return trace.get_tracer(INSTRUMENTATION_NAME)


def run_span(run_id: str, workflow: str) -> AbstractContextManager[Span]:
    """Open the span every node span in this run will hang under."""
    return tracer().start_as_current_span(
        f"dagent.run {workflow}",
        attributes={RUN_ID: run_id, WORKFLOW: workflow},
    )


def node_span(run_id: str, node_id: str, agent: str, attempt: int) -> AbstractContextManager[Span]:
    """Open the span for one node attempt.

    Per *attempt*, not per node: a node that was retried three times took three different
    amounts of time for three different reasons, and averaging them into one span throws
    away the thing you opened the trace to find.
    """
    return tracer().start_as_current_span(
        f"dagent.node {node_id}",
        attributes={RUN_ID: run_id, NODE_ID: node_id, AGENT: agent, ATTEMPT: attempt},
    )
