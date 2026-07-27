"""The diamond: A fans out to B and C, which fan back in to D.

The smallest workflow that exercises everything the Phase 2 executor does — dependency
resolution, concurrent dispatch, fan-in, and output plumbing along the edges.

    uv run python examples/diamond.py

Defined in Python because the YAML loader (SPEC FR-1) has no owning phase yet.
"""

from __future__ import annotations

import asyncio
import json

from dagent.agents import FakeAgent
from dagent.graph import WorkflowBuilder
from dagent.models import Workflow
from dagent.runtime import AgentRegistry, Executor
from dagent.store import InMemoryStateStore


def build() -> Workflow:
    """A -> B, A -> C, B + C -> D."""
    return (
        WorkflowBuilder("diamond")
        .add_node("a", "fake")
        .add_node("b", "fake", inputs={"upstream": "a"})
        .add_node("c", "fake", inputs={"upstream": "a"})
        .add_node("d", "fake", inputs={"left": "b", "right": "c"})
        .build()
    )


async def main() -> None:
    """Run the diamond and print what each node produced."""
    registry = AgentRegistry()
    registry.add("fake", FakeAgent)
    store = InMemoryStateStore()

    workflow = build()
    run = await Executor(registry=registry, store=store).run(workflow, run_id="demo-diamond")

    print(f"run {run.run_id}: {run.state}")
    for node_id in sorted(run.nodes):
        print(f"  {node_id}: {run.nodes[node_id].state}")
    print()
    for node_id in sorted(run.nodes):
        output = await store.load_output(run.run_id, node_id)
        print(f"{node_id} -> {json.dumps(output)}")


if __name__ == "__main__":
    asyncio.run(main())
