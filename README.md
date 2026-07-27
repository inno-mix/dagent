# Dagent

A durable, async DAG execution engine for multi-agent AI workflows — a miniature
Temporal/Airflow where the unit of work is an LLM agent.

Dagent takes a workflow defined as a directed acyclic graph, validates it before anything
runs, executes independent nodes concurrently on `asyncio`, resolves data dependencies
along the edges, enforces retry/timeout/budget policy, and persists state so a crashed run
resumes instead of restarting. The engine is model-agnostic and framework-free: the core
packages import no model SDK, and there is no orchestration framework anywhere.

See [`docs/SPEC.md`](docs/SPEC.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Status

Under construction, phase by phase, against [`docs/ROADMAP.md`](docs/ROADMAP.md).

- **Phase 0 — Scaffold.** Done.
- **Phase 1 — Workflow model + validation.** Done: frozen `Workflow`/`Node`/`Policy`
  schemas, the `NodeState`/`RunState` records, cycle detection that reports the offending
  path, input-satisfaction and unknown-agent checks, `ready_set`, and a typed builder.
- **Phase 2 — In-memory async executor.** Next.

## Quick start

```bash
uv sync
uv run pytest
```

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

## Development

```bash
uv sync                      # install into .venv
uv run pytest                # full suite — no test touches the network
uv run ruff check .          # lint
uv run ruff format .         # format (never hand-format)
uv run mypy dagent           # type check — must be clean
```

All three must pass before any task is complete. See [`docs/AGENTS.md`](docs/AGENTS.md).
