"""Reconstructing a run from the store — FR-10's run inspector.

Everything a run did is already durable; this assembles it into one JSON document. Nothing
here computes anything the engine did not already record, which is the point: if the
inspector can rebuild a run, the store held enough to resume it, and if it cannot, Phase
5's guarantee was never real.

Reads through the ``StateStore`` protocol like everything else, so ``dagent inspect`` works
against Postgres exactly as it works against memory.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from dagent.errors import StoreError
from dagent.models.state import NodeState, RunStateRecord
from dagent.store.base import StateStore

__all__ = ["inspect_run"]


async def inspect_run(store: StateStore, run_id: str, *, outputs: bool = True) -> dict[str, Any]:
    """Return everything the store knows about a run, as JSON-serializable data.

    Args:
        store: Where the run lives.
        run_id: Which run.
        outputs: Include each node's output. Off is useful when the outputs are large and
            the question is about timing or state.

    Returns:
        A mapping with the run's own state, its definition, every node's state and
        timings, the model calls made, and — optionally — each node's output.

    Raises:
        StoreError: If there is no such run.
    """
    run = await store.load_run(run_id)

    return {
        "run": _run(run),
        "workflow": await _workflow(store, run_id),
        "nodes": await _nodes(store, run, outputs=outputs),
        "model_calls": await _model_calls(store, run_id),
    }


def _run(run: RunStateRecord) -> dict[str, Any]:
    """The run-level summary, with the duration worked out rather than left to the reader."""
    return {
        "run_id": run.run_id,
        "workflow_name": run.workflow_name,
        "state": run.state.value,
        "created_at": _moment(run.created_at),
        "updated_at": _moment(run.updated_at),
        "duration_s": _elapsed(run.created_at, run.updated_at),
        "node_count": len(run.nodes),
        "states": _tally(run),
    }


async def _workflow(store: StateStore, run_id: str) -> dict[str, Any] | None:
    """The definition as executed — expanded, if a planner grew it.

    Returned as data rather than omitted, because a dynamically grown run is exactly the
    one whose shape nobody can look up in a file.
    """
    try:
        workflow = await store.load_workflow(run_id)
    except StoreError:
        return None
    return {
        "name": workflow.name,
        "nodes": [
            {
                "id": node.id,
                "agent": node.agent,
                "depends_on": list(node.depends_on),
                "inputs": dict(node.inputs),
                "params": dict(node.params),
            }
            for node in workflow.nodes
        ],
    }


async def _nodes(store: StateStore, run: RunStateRecord, *, outputs: bool) -> list[dict[str, Any]]:
    """Every node's state, timing, attempt count and — where it produced one — output."""
    reported: list[dict[str, Any]] = []
    for node_id in sorted(run.nodes):
        record = run.nodes[node_id]
        node: dict[str, Any] = {
            "node_id": node_id,
            "state": record.state.value,
            "attempt": record.attempt,
            "started_at": _moment(record.started_at),
            "finished_at": _moment(record.finished_at),
            "duration_s": _elapsed(record.started_at, record.finished_at),
            "error": record.error,
        }
        if outputs and record.state is NodeState.SUCCESS:
            # SUCCESS guarantees an output exists — the executor writes it first — so a
            # StoreError here would be a real inconsistency and is not swallowed.
            node["output"] = await store.load_output(run.run_id, node_id)
        reported.append(node)
    return reported


async def _model_calls(store: StateStore, run_id: str) -> list[dict[str, Any]]:
    """Every model call, keyed the way a replay will look them up."""
    return [
        {
            "node_id": call.node_id,
            "attempt": call.attempt,
            "sequence": call.sequence,
            "provider": call.response.provider,
            "model": call.response.model,
            "input_tokens": call.response.input_tokens,
            "output_tokens": call.response.output_tokens,
            "total_tokens": call.response.total_tokens,
            "started_at": _moment(call.started_at),
            "finished_at": _moment(call.finished_at),
            "prompt": call.request.prompt,
            "text": call.response.text,
        }
        for call in await store.load_model_calls(run_id)
    ]


def _tally(run: RunStateRecord) -> dict[str, int]:
    """How many nodes ended in each state — the one line that answers "what happened"."""
    counts: dict[str, int] = {}
    for record in run.nodes.values():
        counts[record.state.value] = counts.get(record.state.value, 0) + 1
    return dict(sorted(counts.items()))


def _moment(value: datetime | None) -> str | None:
    """ISO-8601, or ``None`` where the engine never recorded a time."""
    return value.isoformat() if value is not None else None


def _elapsed(start: datetime | None, end: datetime | None) -> float | None:
    """Seconds between two recorded moments, or ``None`` if either is missing."""
    if start is None or end is None:
        return None
    return (end - start).total_seconds()
