"""How much does the engine itself cost?

SPEC's performance target is the question this answers: *"overhead of the engine itself
(excluding model latency) under a few milliseconds per node at 100 concurrent nodes on a
laptop."* So the agent here does nothing at all. Every microsecond measured is scheduling,
validation, state transitions, store writes and instrumentation — the parts a real run
pays on top of waiting for a model.

An async driver rather than Locust or k6: those measure a service over HTTP, and dagent is
a library. The thing under load is the run loop, and the honest way to load it is to hand
it graphs and time them.

    uv run python benchmarks/load.py                  # the standard sweep
    uv run python benchmarks/load.py --nodes 2000     # one size, in detail
    uv run python benchmarks/load.py --latency-ms 50  # with simulated model latency
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from dagent.graph.builder import WorkflowBuilder
from dagent.models.state import NodeOutput, RunState
from dagent.models.workflow import Workflow
from dagent.runtime.agent import AgentContext
from dagent.runtime.executor import Executor
from dagent.runtime.registry import AgentRegistry
from dagent.store.memory import InMemoryStateStore

SHAPES = ("wide", "chain", "layered")


class Idle:
    """Does nothing, as fast as possible.

    The measurement depends on this being genuinely free: anything it did would be
    attributed to the engine. It does not even yield, because a node that suspends is a
    node whose scheduling cost gets mixed in with the event loop's.
    """

    async def run(self, ctx: AgentContext) -> NodeOutput:
        return None


class Waiting:
    """Sleeps for a fixed time, standing in for a model call."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    async def run(self, ctx: AgentContext) -> NodeOutput:
        await asyncio.sleep(self._seconds)
        return None


def wide(nodes: int) -> Workflow:
    """One source, `nodes - 2` independent workers, one node fanning them all back in.

    The shape SPEC's "100 concurrent nodes" describes, and the one that stresses dispatch:
    every worker is ready at once, so the ready set is as large as it ever gets.
    """
    builder = WorkflowBuilder("wide").add_node("source", "idle")
    workers = [f"w{index}" for index in range(max(nodes - 2, 1))]
    for worker in workers:
        builder.add_node(worker, "idle", depends_on=["source"])
    builder.add_node("sink", "idle", depends_on=workers)
    return builder.build()


def chain(nodes: int) -> Workflow:
    """A single dependency chain — no concurrency at all.

    The opposite extreme: one node ready at a time, so the loop runs its full length once
    per node. This is where any per-pass cost that scales with graph size shows up.
    """
    builder = WorkflowBuilder("chain").add_node("n0", "idle")
    for index in range(1, nodes):
        builder.add_node(f"n{index}", "idle", depends_on=[f"n{index - 1}"])
    return builder.build()


def layered(nodes: int, width: int = 10) -> Workflow:
    """Alternating fan-out and fan-in — the shape a real workflow tends to have."""
    builder = WorkflowBuilder("layered")
    previous: list[str] = []
    made = 0
    layer = 0
    while made < nodes:
        current: list[str] = []
        for index in range(min(width, nodes - made)):
            node_id = f"l{layer}_{index}"
            builder.add_node(node_id, "idle", depends_on=previous)
            current.append(node_id)
            made += 1
        previous = current
        layer += 1
    return builder.build()


BUILDERS: dict[str, Callable[[int], Workflow]] = {
    "wide": wide,
    "chain": chain,
    "layered": layered,
}


@dataclass(frozen=True, slots=True)
class Result:
    """One measurement."""

    shape: str
    nodes: int
    seconds: float

    @property
    def per_node_ms(self) -> float:
        """Engine overhead attributed to each node."""
        return self.seconds * 1000 / self.nodes


async def once(workflow: Workflow, agent: Callable[[], object], run_id: str) -> float:
    """Run one workflow and return how long the engine took."""
    registry = AgentRegistry()
    registry.add("idle", agent)  # type: ignore[arg-type]
    executor = Executor(registry=registry, store=InMemoryStateStore())

    start = time.perf_counter()
    run = await executor.run(workflow, run_id=run_id)
    elapsed = time.perf_counter() - start

    if run.state is not RunState.SUCCEEDED:
        raise SystemExit(f"benchmark run did not succeed: {run.state}")
    return elapsed


async def measure(shape: str, nodes: int, *, repeat: int, latency_ms: float) -> Result:
    """Build one graph and time it `repeat` times, reporting the median."""
    workflow = BUILDERS[shape](nodes)
    agent: Callable[[], object] = Idle if latency_ms <= 0 else (lambda: Waiting(latency_ms / 1000))

    # One untimed pass first: the first run of a shape pays for import-time caches and a
    # cold allocator, and reporting that as the engine's cost would be a lie.
    await once(workflow, agent, "warmup")
    timings = [await once(workflow, agent, f"r{index}") for index in range(repeat)]
    return Result(shape, len(workflow.nodes), statistics.median(timings))


def report(results: Sequence[Result], *, latency_ms: float) -> None:
    """Print the table, and say plainly whether the target was met."""
    print(f"\n{'shape':<9} {'nodes':>6} {'total (s)':>10} {'per node (ms)':>14}")
    print("-" * 43)
    for result in results:
        print(
            f"{result.shape:<9} {result.nodes:>6} "
            f"{result.seconds:>10.4f} {result.per_node_ms:>14.4f}"
        )

    if latency_ms > 0:
        print("\n(with simulated model latency; per-node figures include the sleep)")
        return

    at_target = [r for r in results if r.shape == "wide" and 90 <= r.nodes <= 110]
    if at_target:
        worst = max(r.per_node_ms for r in at_target)
        verdict = "MET" if worst < 3 else "MISSED"
        print(
            f"\nSPEC target — under a few ms per node at 100 concurrent nodes: {verdict} "
            f"({worst:.4f} ms/node)"
        )


async def main() -> None:
    """Run the sweep the README quotes, or whatever the caller asked for instead."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=None, help="One size instead of the sweep.")
    parser.add_argument("--shape", choices=SHAPES, default=None, help="One shape.")
    parser.add_argument("--repeat", type=int, default=5, help="Timed runs per point.")
    parser.add_argument(
        "--latency-ms", type=float, default=0.0, help="Simulated per-node model latency."
    )
    args = parser.parse_args()

    shapes = [args.shape] if args.shape else list(SHAPES)
    sizes = [args.nodes] if args.nodes else [10, 100, 500, 1000, 2000]

    results = [
        await measure(shape, size, repeat=args.repeat, latency_ms=args.latency_ms)
        for shape in shapes
        for size in sizes
    ]
    report(results, latency_ms=args.latency_ms)


if __name__ == "__main__":
    asyncio.run(main())
