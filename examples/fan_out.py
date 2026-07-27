"""A wide fan-out: one source, N independent branches, one sink that merges them.

The shape the research workflow will take once the planner decides N at runtime
(Phase 6). Here N is fixed, which is the point — Phase 2 runs static graphs only.

    uv run python examples/fan_out.py
"""

from __future__ import annotations

import asyncio

from dagent.agents import FakeAgent
from dagent.graph import WorkflowBuilder, ready_set
from dagent.models import NodeState, Workflow
from dagent.runtime import AgentRegistry, Executor
from dagent.store import InMemoryStateStore

BRANCHES = 5


def build(branches: int = BRANCHES) -> Workflow:
    """One source, ``branches`` parallel workers, one sink reading all of them."""
    builder = WorkflowBuilder("fan-out").add_node("plan", "fake")
    for index in range(branches):
        builder.add_node(f"work_{index}", "fake", inputs={"topic": "plan"})
    return builder.add_node(
        "merge",
        "fake",
        inputs={f"branch_{index}": f"work_{index}" for index in range(branches)},
    ).build()


async def main() -> None:
    """Show the ready set widening, then run the workflow."""
    workflow = build()

    states = {node.id: NodeState.PENDING for node in workflow.nodes}
    print("ready at start :", ready_set(workflow, states))
    states["plan"] = NodeState.SUCCESS
    print("after plan     :", ready_set(workflow, states), "<- dispatched together")

    registry = AgentRegistry()
    registry.add("fake", FakeAgent)
    run = await Executor(registry=registry, store=InMemoryStateStore()).run(
        workflow, run_id="demo-fan-out"
    )

    print(f"\nrun {run.run_id}: {run.state} across {len(run.nodes)} nodes")


if __name__ == "__main__":
    asyncio.run(main())
