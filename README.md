[![CI](https://github.com/inno-mix/dagent/actions/workflows/ci.yml/badge.svg)](https://github.com/inno-mix/dagent/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/inno-mix/dagent)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

# Dagent

A durable, async DAG execution engine for multi-agent AI workflows — a miniature
Temporal/Airflow where the unit of work is an LLM agent.

Dagent takes a workflow defined as a directed acyclic graph, validates it before anything
runs, executes independent nodes concurrently on `asyncio`, resolves data dependencies
along the edges, enforces retry/timeout/budget policy, and persists state so a crashed run
resumes instead of restarting. The engine is model-agnostic and framework-free: the core
packages import no model SDK, and there is no orchestration framework anywhere — that
boundary is enforced mechanically by [`tests/test_isolation.py`](tests/test_isolation.py),
which fails the build if a core module ever imports one.

## What it does

- **Defines workflows as data.** A `Workflow` is a frozen, immutable set of `Node`s, each
  bound to a registered agent, with declared inputs and an optional retry/timeout policy.
- **Validates before executing.** Cycles, unknown agents, and inputs that read from a node
  the reader never waits for are all rejected at submit time, with the offending path
  named in the error.
- **Schedules concurrently.** Independent nodes run in parallel on `asyncio`; completing a
  node recomputes the ready set so fan-out and fan-in fall out of the same loop.
- **Stays durable.** Node state and outputs persist through a storage-agnostic
  `StateStore`, so a crashed run resumes from its last checkpoint instead of restarting.
- **Stays replayable.** Every source of non-determinism — the clock, model calls — is
  injected, never called directly from core logic, so a recorded run can be replayed
  offline for debugging and tests.

See [`docs/SPEC.md`](docs/SPEC.md) for the full functional specification,
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the system design and the decision
records behind it, and [`docs/ROADMAP.md`](docs/ROADMAP.md) for how the build is
sequenced phase by phase.

## Status

Under construction, phase by phase, against [`docs/ROADMAP.md`](docs/ROADMAP.md).

| Phase | Goal | Status |
| --- | --- | --- |
| 0 — Scaffold | Repo lints, type-checks, and tests from an empty shell | Done |
| 1 — Workflow model + validation | Frozen `Workflow`/`Node`/`Policy` schemas, cycle detection, input-satisfaction checks, `ready_set`, typed builder | Done |
| 2 — In-memory async executor | Run a fixed DAG concurrently, passing outputs along edges | Done |
| 3 — Agent plugin interface | Real LLM agents behind a model-agnostic contract | Done |
| 4 — Policy engine | Retries, timeouts, budgets, failure semantics | Done |
| 5 — Persistence + crash resume | Kill it mid-run, restart, get the same result | Done |
| 6 — Dynamic DAG expansion | A planner agent grows the graph at runtime | Done |
| 7 — Observability + run inspector | Tracing, metrics, and a CLI `inspect` command | Done |
| 8 — Capstone: distributed workers | Same core, Redis Streams transport, Postgres store | Next |

## Performance, and the bottleneck it found

SPEC's target is that the engine's own overhead — everything that is not waiting for a
model — stays "under a few milliseconds per node at 100 concurrent nodes on a laptop."
`benchmarks/load.py` measures exactly that, with an agent that does nothing at all, so
every microsecond it reports belongs to scheduling, validation, state transitions, store
writes and instrumentation.

```
uv run python benchmarks/load.py

shape      nodes  total (s)  per node (ms)
-------------------------------------------
wide         100     0.0048         0.0475     <- the target: 60x under budget
wide        2000     0.1403         0.0702
chain        100     0.0114         0.1140
chain       2000     1.1346         0.5673     <- the pathological shape
layered     2000     0.4250         0.2125
```

The target is met with room to spare, and with 50 ms of simulated model latency the
engine disappears entirely: 100 nodes finish in 166 ms — three sequential waves of 50 ms —
and 500 nodes finish in the same wall time, which is what "runs independent nodes
concurrently" is supposed to mean.

**The bottleneck was `ready_set`, at 82% of engine time.** Profiling a 2,000-node chain
showed it recomputing the entire ready set on every pass of the run loop: 2,001 passes ×
2,000 nodes, two million `frozenset` allocations, six million dictionary lookups. Two
things were wrong. The loop was rebuilding its node index, its declaration-order map and
every node's edge set on each pass, none of which change unless the graph does — so those
are now derived once per graph *version*, held against the `Workflow` object itself and
invalidated by identity when an expansion replaces it. That took the pathological case
from 1.64 s to 1.13 s, a 31% cut, and left the target case unchanged because it was never
the problem there.

What remains is algorithmic and worth naming precisely rather than hiding. Recomputing
readiness from scratch after each completion is O(V+E) per pass, so a graph that completes
one node at a time costs O(V²) overall — visible above as `chain`'s per-node figure rising
with size while `wide`'s stays flat. It does not breach the target at any size measured,
and it only bites on graphs with no concurrency, which is the shape this engine is least
useful for. The fix, when a workload demands it, is the standard one: keep a count of each
node's unfinished dependencies and decrement it on completion, turning the whole run into
O(V+E) instead of O(V+E) per pass. That is deliberately not done yet — it is surgery on
the most delicate code in the project to serve a workload SPEC does not describe, and
`ready_set` staying a pure function of the graph and the states is worth more today than
the constant factor.

## Architecture at a glance

```
                         submit(workflow)
                               │
                        ┌──────▼───────┐
                        │  Validator   │  cycles, input satisfaction, agent exists
                        └──────┬───────┘
                               │ frozen, valid Workflow
                        ┌──────▼───────┐        ┌──────────────┐
                        │   Executor   │◀──────▶│  StateStore  │  durable NodeState
                        │  (scheduler) │        └──────────────┘  + outputs
                        └──────┬───────┘
              ready set        │  acquires global + per-provider permits
        ┌──────────────┬───────┴───────┬──────────────┐
   ┌────▼────┐    ┌────▼────┐     ┌────▼────┐    ┌────▼────┐
   │  Agent  │    │  Agent  │ ... │  Agent  │    │ Planner │─┐ emits new nodes
   └────┬────┘    └────┬────┘     └────┬────┘    └────┬────┘ │ (dynamic expansion)
        │ output       │               │              │      │
        └──────────────┴───────┬───────┴──────────────┘      │
                               │  policy: retry / timeout / budget
                        ┌──────▼───────┐                      │
                        │ Observability│  spans, metrics, logs│
                        └──────────────┘◀─────────────────────┘
```

Everything the executor does is a loop: compute the ready set → acquire concurrency
permits → dispatch → on completion persist state and recompute the ready set → repeat
until the graph is terminal. Full component-by-component detail, the hard problems
(durable resume, dynamic expansion, backpressure, deterministic replay), and the v1→v2
scaling path live in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/inno-mix/dagent.git
cd dagent
uv sync                # installs into .venv from pyproject.toml / uv.lock
cp .env.example .env   # then fill in provider keys once Phase 3 needs them
```

## Quick start

Run the test suite to confirm the install works:

```bash
uv run pytest
```

Build and validate a workflow (currently the only executable surface — the async
executor lands in Phase 2):

```python
from dagent.graph import WorkflowBuilder, ready_set
from dagent.models import NodeState

workflow = (
    WorkflowBuilder("diamond")
    .add_node("plan", "fake")
    .add_node("research_a", "fake", inputs={"topic": "plan"})
    .add_node("research_b", "fake", inputs={"topic": "plan"})
    .add_node("synthesize", "fake", inputs={"a": "research_a", "b": "research_b"})
    .build()
)

states = {node.id: NodeState.PENDING for node in workflow.nodes}
ready_set(workflow, states)  # ("plan",)
```

An `inputs` entry both feeds a node and makes it wait: the builder adds each input's
source to that node's dependencies, and validation rejects any input that reads from a
node the reader does not wait for, because that is a race.

The `dagent` CLI is also installed as a script:

```bash
uv run dagent version
```

## Core concepts

| Concept | What it is |
| --- | --- |
| **Workflow** | An immutable definition: a set of nodes plus their data dependencies. Validated once at submission. |
| **Node** | A named unit of work bound to a registered agent, with declared inputs (which upstream outputs it consumes) and an optional policy override. |
| **Agent** | The pluggable behavior behind a node — an LLM call, a tool, or pure logic — implementing a single async contract. |
| **Run** | One execution of a workflow, tracked by a `RunState` and a per-node `NodeState`. |
| **Context** | What an agent receives at execution: its resolved inputs, run metadata, the injected clock, the model client, and the budget handle. |

## Project layout

```
dagent/
  models/        # frozen pydantic workflow + state schemas (no logic)
  graph/         # validation (cycles, input satisfaction), topo/ready-set, typed builder
  runtime/       # agent contract, registry, the async executor
  policy/        # retry/backoff, timeouts, concurrency + budget limits
  store/         # StateStore protocol + memory/postgres impls
  agents/        # concrete LLM agents (planner/researcher/synthesizer/critic)
  observability/ # tracing, metrics, structured logging setup
  errors.py      # exception hierarchy
  cli.py         # typer entrypoint
tests/           # mirrors the package layout
```

## Development

```bash
uv sync                      # install into .venv
uv run pytest                # full suite — no test touches the network
uv run pytest tests/graph -q # scope to one package while iterating
uv run ruff check .          # lint
uv run ruff format .         # format (never hand-format)
uv run mypy dagent           # type check — must be clean
```

All three (`pytest`, `ruff check`, `mypy`) must pass before any task is considered
complete. See [`docs/AGENTS.md`](docs/AGENTS.md) for the full set of conventions this
project is built against — module boundaries, testing rules, and commit style.

## Documentation

- [`docs/SPEC.md`](docs/SPEC.md) — functional and non-functional requirements.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design and decision records.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — the phased build plan and acceptance criteria.
- [`docs/AGENTS.md`](docs/AGENTS.md) — contributor/agent operating instructions.

## License

MIT — see [`LICENSE`](LICENSE) for details.
