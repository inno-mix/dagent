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

- **Phase 0 — Scaffold.** Done: the repo lints, type-checks, and tests from an empty
  shell, with the package skeleton and the error hierarchy in place.
- **Phase 1 — Workflow model + validation.** Next.

## Quick start

```bash
uv sync
uv run pytest
```

## Development

```bash
uv sync                      # install into .venv
uv run pytest                # full suite — no test touches the network
uv run ruff check .          # lint
uv run ruff format .         # format (never hand-format)
uv run mypy dagent           # type check — must be clean
```

All three must pass before any task is complete. See [`docs/AGENTS.md`](docs/AGENTS.md).
