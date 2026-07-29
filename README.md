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
| 8 — Capstone: distributed workers | Same core, Redis Streams transport, Postgres store | Done |

## Distributed workers — and what did *not* change

The last phase moves execution off the coordinator's event loop and onto a pool of
interchangeable worker processes, talking over Redis Streams, with Postgres as the shared
store. The interesting part of that sentence is how little of the engine it touched.

```
  coordinator process                          worker process  (x N)
  ┌────────────────────────┐                   ┌────────────────────────┐
  │ validate               │   dagent:work     │ claim ─┐               │
  │ ready_set  ────────────┼──── stream ──────▶│        ▼               │
  │ failure modes          │  (consumer group) │  NodeRunner            │
  │ RunGraph (one owner)   │                   │   retry / timeout      │
  │ derive RunState        │◀─── results ──────┤   permits / budget     │
  └──────────┬─────────────┘  stream per run   └──────────┬─────────────┘
             │                                            │
             └──────────────▶ Postgres StateStore ◀────────┘
                       state, outputs, model calls, definition
```

**What changed: one seam, two implementations.** The executor's loop used to call
`asyncio.create_task` inline. It now calls a `Dispatcher` — `LocalDispatcher` spawns a task
on this loop, `QueueDispatcher` posts a message and waits. Everything that made a node run
correctly moved to `runtime/node.py` as `NodeRunner`, unchanged, so a worker gets the retry
loop, the timeout, the dual-axis permits, the budget metering, the recording wrapper and
the write ordering by construction rather than by reimplementation. `executor.py` went from
720 lines doing two jobs to a scheduler that does one.

**What did not change, and this is the payoff.** `dagent/graph` was not touched: validation,
cycle detection, `ready_set`, `descendants`, `expand_workflow` are the same pure functions,
still with no async and no I/O. `dagent/policy` was not touched. The three failure modes,
the budget, the backoff and jitter, the idempotency key, `resume`, and the run-state
derivation are all shared code. There is exactly one run loop, and a distributed run and a
single-process run go through it identically — which is what
`test_a_distributed_run_produces_exactly_what_a_single_process_run_produces` asserts,
node state by node state and output by output.

**At-least-once is safe because Phase 5 made it safe.** A work message *is* the idempotency
key `(run_id, node_id, attempt)` and carries nothing else — not the node, not its inputs,
because those live in the store and a message carrying them would be a second source of
truth that disagrees the first time a planner expands anything. A worker killed mid-node
leaves its message pending in the consumer group; another worker's `XAUTOCLAIM` takes it
over, keeping the message id, and re-runs it *at the same attempt*. So the node comes back
under the key the outside world has already seen, and an agent that deduplicates on
`ctx.idempotency_key` commits exactly once. That is DR-4 being cashed in: the investment
was made four phases before the feature that needed it.

Verified against real infrastructure rather than asserted — Redis 7.4, Postgres 16, three
OS processes, `kill -9` on the one holding a node:

```
=== every execution of the agent body ===        === effects actually committed ===
kd2:slow_a:0 pid=35741 started                   kd2_slow_a_0
kd2:slow_a:0 pid=35741 committed   <- then SIGKILL, before it could report
kd2:slow_a:0 pid=35742 started     <- taken over, same attempt
kd2:slow_a:0 pid=35742 already committed         kd2_slow_b_0
...                                              kd2_slow_c_0
coordinator run=kd2 state=succeeded              (4 executions, 3 commits)
```

**Three things genuinely got harder, stated rather than glossed over.**

*The graph still has exactly one owner.* A planner can run on any worker, but a worker
cannot merge its own expansion: the copy of the workflow it loaded may already be stale,
because another worker's planner may have grown it since. So the request rides home on the
result and the coordinator — the one process looking at a graph nobody else is editing —
validates and merges it. A rejected expansion still fails only the node that asked, one hop
later than before. Results are acknowledged *after* being merged, not on receipt, so a
coordinator that dies holding an unmerged expansion gets the report again on resume. That
window was a real bug, found by a test written from the requirement: `resume` had nothing
outstanding, concluded the run was finished, and reported success having silently dropped
the fan-out. `Dispatcher.start` now returns what the last coordinator never acknowledged.

*`fail_fast` degrades honestly.* A coroutine in another process cannot be cancelled, so the
mode stops dispatching and waits for the siblings already out there, then records what they
actually did. Writing "cancelled" for a node a worker was still running would put a
falsehood in the store to preserve a word.

*A token budget is now per worker, not per run.* Nothing sums spending across processes.
Concurrency caps are per worker by design — each process gets its own slice of the
provider's rate limit — but a run-wide ceiling would need the coordinator to admit calls,
and that is a round trip per model call. Documented, not pretended.

**One bug only distribution could find.** `PostgresStateStore.connect` ran
`CREATE TABLE IF NOT EXISTS`, which looks like a concurrency guarantee and is not: two
connections that both find a table missing both try to create it, and the loser gets a
unique-violation from Postgres' catalogue. Invisible while exactly one process ever
connected; it fired on the first real two-worker run, when the pool came up together. Now
serialized on an advisory lock, with a test that opens the store eight times at once.

```bash
# terminal 1..N — workers are interchangeable and none is in charge
uv run --env-file .env dagent worker --queue redis --store postgres

# terminal 0 — the coordinator: validates, schedules, owns the graph, runs no agents
uv run --env-file .env dagent run examples/research_dynamic.yaml \
  --run-id live-1 --queue redis --store postgres
```

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
until the graph is terminal. That is the single-process shape; the distributed one differs
at exactly one arrow, and is drawn above under
[Distributed workers](#distributed-workers--and-what-did-not-change). Full
component-by-component detail and the hard problems (durable resume, dynamic expansion,
backpressure, deterministic replay) live in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/inno-mix/dagent.git
cd dagent
uv sync                # installs into .venv from pyproject.toml / uv.lock
cp .env.example .env   # then fill in provider keys
```

Nothing above needs a database or a broker: a single-process run uses neither. The two
extras are opt-in, on the same terms — `uv sync --extra postgres` for durable state,
`--extra redis` for distributed workers.

## Quick start

Run the test suite to confirm the install works:

```bash
uv run pytest
```

Build a workflow and run it. No provider needed — `fake` agents do no I/O:

```python
import asyncio

import dagent.agents  # registers the built-in agents by name; the core never imports them
from dagent.graph import WorkflowBuilder
from dagent.runtime import Executor, default_registry
from dagent.store import InMemoryStateStore

workflow = (
    WorkflowBuilder("diamond")
    .add_node("plan", "fake")
    .add_node("research_a", "fake", inputs={"topic": "plan"})
    .add_node("research_b", "fake", inputs={"topic": "plan"})
    .add_node("synthesize", "fake", inputs={"a": "research_a", "b": "research_b"})
    .build()
)

run = asyncio.run(
    Executor(registry=default_registry, store=InMemoryStateStore()).run(workflow, run_id="r1")
)
print(run.state)  # RunState.SUCCEEDED — with research_a and research_b genuinely concurrent
```

An `inputs` entry both feeds a node and makes it wait: the builder adds each input's
source to that node's dependencies, and validation rejects any input that reads from a
node the reader does not wait for, because that is a race.

From the command line, against a real provider:

```bash
uv run --env-file .env dagent run examples/research_dynamic.yaml --run-id r1
```

`research_dynamic.yaml` writes down two nodes. The planner reads the question, decides how
many researchers the answer needs, and grows the graph at run time — which is why the
definition is persisted with the run rather than reloaded from the file.

Point it at Postgres and the run outlives the process, which is what `resume` and `inspect`
need — a run whose state died with the process that produced it is not a run you can ask
about afterwards:

```bash
export DAGENT_POSTGRES_DSN=postgresql://...
uv run --env-file .env dagent run examples/research_dynamic.yaml --run-id r2 --store postgres
uv run dagent resume r2      # if that was interrupted: continue it, at the same attempt
uv run dagent inspect r2     # every node, timing, output and model call, as JSON
```

Same workflow across machines — one coordinator, as many workers as you like:

```bash
export DAGENT_REDIS_URL=redis://...
uv run --env-file .env dagent worker --queue redis --store postgres    # in each worker
uv run --env-file .env dagent run examples/research_dynamic.yaml \
  --run-id r3 --queue redis --store postgres                           # the coordinator
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
  runtime/       # agent contract, registry, the coordinator, the node runner, the worker
  policy/        # retry/backoff, timeouts, concurrency + budget limits
  store/         # StateStore protocol + memory/postgres impls
  transport/     # WorkQueue protocol + memory/redis impls (the v2 work channel)
  agents/        # concrete LLM agents (planner/researcher/synthesizer/critic)
  observability/ # tracing, metrics, structured logging setup
  errors.py      # exception hierarchy
  loader.py      # YAML workflow files -> Workflow
  cli.py         # typer entrypoint
tests/           # mirrors the package layout
benchmarks/      # the load driver behind the performance section
```

Inside `runtime/`, the split that made Phase 8 possible: `executor.py` schedules a graph,
`node.py` runs a single node, `dispatch.py` is the seam between them, and `worker.py` is
`node.py` with a queue in front of it.

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
