# AGENTS.md

Operating instructions for any AI coding agent working in this repository.
This file is the contract. Follow it literally. When something here conflicts
with your own assumptions, this file wins. When this file is silent, prefer the
conventions already present in the codebase, then ask.

> **Project:** Dagent — a durable, async DAG execution engine for multi-agent AI workflows.
> **One-line mental model:** a miniature Temporal/Airflow where the unit of work is an LLM agent.

---

## 1. What you are building

Read `SPEC.md` for the *what* and `ARCHITECTURE.md` for the *how*. Do not start
coding until you have read both. Pick work from `ROADMAP.md` — phases are ordered
and each has explicit acceptance criteria. Do not skip ahead; later phases assume
earlier ones are complete and tested.

The core insight to preserve at all times: **the interesting engineering is the
concurrent, durable execution engine, not the prompts.** Never let agent/prompt
code leak into the scheduler, and never let scheduler concerns leak into agents.

## 2. Golden rules

1. **Tests come with the code, not after.** Every new module lands with unit
   tests in the same PR. No test, no merge.
2. **Never widen a public interface to make a test pass.** Fix the test or the design.
3. **The core has no LLM dependency.** `dagent/graph`, `dagent/runtime/executor.py`,
   `dagent/policy`, and `dagent/store` must import zero model SDKs. Agents depend on
   the core; the core never depends on agents. If you're tempted to `import openai`
   in the executor, stop — you've made a design error.
4. **Determinism is sacred.** All non-determinism (time, randomness, model I/O)
   goes through injectable seams so runs can be replayed. Never call
   `time.time()`, `random`, or a model client directly inside core logic.
5. **Idempotency before durability.** A node may be executed more than once after a
   crash. Writing resume logic before node execution is idempotent is a bug factory.
   Get the idempotency key design right first (see ARCHITECTURE §"Durable execution").
6. **Fail loud in dev, degrade gracefully in prod.** Assertions and strict
   validation at submit time; circuit breakers and fallbacks at run time.
7. **No secrets in code, logs, traces, or fixtures.** Model keys come from env only.

## 3. Environment & commands

Python 3.11+. Dependency and env management via `uv`.

```bash
# setup
uv sync                      # install deps into .venv from pyproject/uv.lock
cp .env.example .env         # then fill in provider keys

# the loop you run constantly
uv run pytest -q             # full test suite
uv run pytest tests/graph -q # scope to one package while iterating
uv run ruff check .          # lint
uv run ruff format .         # format (do NOT hand-format)
uv run mypy dagent           # type check — must be clean, no new ignores
uv run dagent run examples/research.yaml   # run a workflow end to end
```

**Definition of done for any task:** `pytest`, `ruff check`, and `mypy` all pass
clean, the new behavior has tests, and any public API change is reflected in the
relevant `.md` doc. Run all three before you claim a task is complete.

## 4. Tech stack (do not add to this without a note in ARCHITECTURE.md)

- **Runtime:** `asyncio` (single-process for v1).
- **Models/validation:** `pydantic` v2 for all workflow, node, and config schemas.
- **Persistence:** `StateStore` protocol with an in-memory impl (v1) and Postgres
  via `asyncpg` (v2). Code against the protocol, never a concrete store.
- **Transport (v2 scaling):** Redis Streams for distributed workers.
- **Observability:** OpenTelemetry (traces + metrics), structured logs via `structlog`.
- **HTTP to providers:** `httpx.AsyncClient`, one shared client, never per-call.
- **CLI:** `typer`.

## 5. Code conventions

- **Async-first.** Public runtime APIs are `async def`. Never block the event loop:
  no `requests`, no sync file I/O in hot paths, no `time.sleep` (use `asyncio.sleep`).
- **Typing is mandatory.** Full annotations on every public function. `mypy` clean.
  Prefer `Protocol` for extension points over inheritance.
- **Errors are typed.** Define a small exception hierarchy in `dagent/errors.py`
  (`DagentError` → `ValidationError`, `PolicyError`, `AgentError`, `StoreError`).
  Never raise bare `Exception`; never `except Exception: pass`.
- **Immutability at boundaries.** Workflow definitions are frozen models. Runtime
  state is mutated only through the `StateStore`, never by reaching into objects.
- **Naming:** `NodeState` (per-node), `RunState` (whole run). Keep the distinction
  crisp — conflating them is the single most common bug in this codebase.
- **Docstrings** on every public symbol: one line on *what*, and for anything
  non-obvious, a line on *why*. Skip docstrings on trivial private helpers.
- **File size smell:** if a module passes ~300 lines, it's probably doing two jobs.

## 6. Testing conventions

- **Unit tests** use a fake agent and the in-memory store — fast, no network.
- **Never call a real model in a test.** Use the recorded-replay fixture; a test
  that hits an API is a broken test.
- **Concurrency tests matter most.** For the executor, assert on: ready-set
  correctness, that per-provider concurrency caps are never exceeded (instrument a
  counter), fan-out/fan-in ordering, and clean cancellation/timeout behavior.
- **Crash-resume tests:** kill the run between two nodes (simulate by raising in a
  seam), reload state from the store, resume, and assert no node's side effect ran
  twice and the final result is identical to the uninterrupted run.
- Property-based tests (`hypothesis`) for graph validation: random DAGs in →
  topo order out, random graphs with a cycle in → `ValidationError` out.
- Target coverage on `graph/`, `runtime/`, `policy/`, `store/` is 90%+. Agents and
  CLI can be lower.

## 7. Boundaries — do NOT do these

- Do **not** introduce a heavyweight framework (LangChain/LlamaGraph/etc.) into the
  core. The whole point is that *you* wrote the engine. A thin provider SDK inside
  an agent module is fine; an orchestration framework anywhere is not.
- Do **not** add a database, message broker, or cloud dependency ahead of the phase
  that calls for it. v1 is single-process and in-memory on purpose.
- Do **not** refactor across module boundaries "while you're in there." Keep PRs
  scoped to one roadmap task.
- Do **not** silently change a public signature used by another module without
  updating callers, tests, and docs in the same change.
- Do **not** commit generated artifacts, `.env`, coverage reports, or `.venv`.

## 8. Working style

- Work in **small, reviewable increments** that map to one roadmap task. State which
  task you're on before you start.
- When a task is ambiguous, state the assumption you're making and proceed; don't
  stall. Flag the assumption in the PR description.
- Leave the campsite cleaner: if you touch a file with an obvious, in-scope bug, fix
  it and note it. Anything out of scope goes in `TODO` comments or an issue, not
  into the current change.
- Prefer boring, explicit code over clever code. This project is read by employers.
  Readability is a feature.

## 9. Commit & PR conventions

- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- One logical change per commit. Message says *why*, not just *what*.
- PR description states: which roadmap task, what changed, how it was tested, and
  any assumption or tradeoff you made.

## 10. Where things live

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
examples/        # sample workflow definitions (yaml)
```

Keep this map accurate. If you move something, update this section and
`ARCHITECTURE.md` in the same PR.
