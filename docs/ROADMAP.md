# ROADMAP.md — Dagent

The build plan, sequenced. Each phase is a set of tasks small enough for one PR, with
explicit **acceptance criteria** an agent can check itself against. Phases are ordered
by dependency — do not start a phase until the previous one's acceptance criteria pass.

A task is **done** only when: its acceptance criteria hold, `pytest` + `ruff check` +
`mypy` are clean, tests were added, and any public API change is reflected in the docs.

---

## Phase 0 — Scaffold
Goal: a repo that lints, type-checks, and tests from an empty shell.

- [ ] `pyproject.toml` (uv), `.env.example`, `ruff` + `mypy` config, `pytest` config.
- [ ] Package skeleton exactly as in AGENTS.md §10, each module a stub.
- [ ] `dagent/errors.py` with the exception hierarchy.
- [ ] CI: one workflow running lint, type-check, tests on push.

**Acceptance:** `uv sync && uv run pytest && uv run ruff check . && uv run mypy dagent`
all succeed on an essentially empty project. CI green.

---

## Phase 1 — Workflow model + validation
Goal: define workflows and reject invalid ones before any execution.

- [ ] Frozen pydantic models: `Workflow`, `Node`, `Policy` (`models/workflow.py`).
- [ ] `NodeState`/`RunState` enums and state records (`models/state.py`).
- [ ] `graph/validate.py`: cycle detection (report the cycle path), input-satisfaction
      check, unknown-agent check.
- [ ] `graph/topo.py`: `ready_set(workflow, states)` and a topological order helper.
- [ ] Typed Python builder for constructing a `Workflow`.

**Acceptance:** property tests pass — random acyclic graphs validate and topo-sort;
random graphs containing a cycle raise `ValidationError` naming the cycle; a node
referencing a non-existent upstream output is rejected. No async, no I/O in this phase.

---

## Phase 2 — In-memory async executor (static DAG)
Goal: run a fixed DAG concurrently, passing outputs along edges.

- [ ] `store/base.py` `StateStore` protocol + `store/memory.py` impl.
- [ ] `runtime/agent.py`: `Agent` protocol + `AgentContext` (with injected `Clock`).
- [ ] `runtime/registry.py`: `@register` decorator.
- [ ] `runtime/executor.py`: the ready-set run loop, fan-out dispatch, fan-in
      recomputation, output resolution into downstream inputs.
- [ ] A `FakeAgent` (no network) and a couple of example static workflows.

**Acceptance:** a diamond DAG (A → B, A → C, B+C → D) runs with B and C concurrent and
D receiving both outputs; final states all `SUCCESS`; a test asserts B and C actually
overlapped in time. Executor imports no model SDK.

---

## Phase 3 — Agent plugin interface + real LLM agents
Goal: two real agents behind the same contract, model-agnostic.

- [ ] Provider-agnostic model client seam (shared `httpx.AsyncClient`), injected via
      `AgentContext`; keys from env only.
- [ ] Two real agents in `dagent/agents` (e.g. `researcher`, `synthesizer`), each
      registered and independently testable with a fake client.
- [ ] Recording wrapper that logs model I/O into the run record (foundation for replay).

**Acceptance:** a workflow of two real agents runs end to end against a live provider
via the CLI; the same workflow runs in tests with the fake client and zero network.
Core packages still import no model SDK — only `agents/` do.

---

## Phase 4 — Policy engine
Goal: retries, timeouts, budgets, and failure semantics.

- [ ] `policy/retry.py`: max attempts, exponential backoff + full jitter, retryable
      classification.
- [ ] Per-node timeout with clean cancellation.
- [ ] `policy/limits.py`: global + per-provider semaphores (ordered acquisition,
      released in `finally`); `Budget` (token/cost ceiling, admission control).
- [ ] Run-level failure semantics: `fail_fast` / `run_to_completion` / `skip_downstream`.

**Acceptance:** a flaky fake agent that fails twice then succeeds is retried to success;
a hanging agent is cancelled at its timeout; with caps set to 2, an instrumented
counter shows in-flight nodes for a provider never exceed 2; exceeding the budget ends
the run in `BUDGET_EXCEEDED` with no further model calls admitted.

---

## Phase 5 — Persistence + crash resume
Goal: kill it mid-run, restart, get the same result.

- [ ] Idempotency key `(run_id, node_id, attempt)` threaded through node execution.
- [ ] State transitions + outputs persisted through the store at each step.
- [ ] `resume(run_id)`: reload state, skip `SUCCESS` nodes, re-dispatch interrupted
      ones safely.
- [ ] `postgres.py` store impl behind the same protocol (`asyncpg`).

**Acceptance:** a test injects a crash between two nodes, reloads from the store,
resumes, and asserts (a) no node's side effect ran twice and (b) the final outputs are
byte-identical to an uninterrupted run. Passes against both memory and Postgres stores.

---

## Phase 6 — Dynamic DAG expansion
Goal: a planner grows the graph at runtime.

- [ ] `planner` agent that emits new node definitions.
- [ ] Executor path to revalidate the augmented graph (stay acyclic) and insert nodes
      as `PENDING`, serialized through the run loop.
- [ ] Guardrails: expansion may not add a dependency that strands a running branch;
      bounded expansion depth.

**Acceptance:** the research workflow — planner fans out to N researchers, then a
synthesizer fans them in — runs end to end with N decided at runtime; a test with a
planner that would introduce a cycle is rejected without deadlocking the run.

---

## Phase 7 — Observability + run inspector
Goal: see inside a run.

- [ ] OpenTelemetry: one span per node nested under a run span, with attempt/provider/
      token attributes.
- [ ] Metrics: in-flight, ready-set size, per-provider concurrency, retries, run
      duration, tokens/cost.
- [ ] `structlog` correlated by `run_id`/`node_id`.
- [ ] CLI `inspect <run_id>` dumping node states, timings, and outputs as JSON.

**Acceptance:** running the research workflow produces a trace whose span tree matches
the DAG; metrics are scrapeable (Prometheus endpoint or exporter); `inspect` output
fully reconstructs what happened in a run.

---

## Phase 8 — Capstone: distributed workers
Goal: the scaling story. Same core, different transport.

- [ ] Executor becomes a coordinator that enqueues ready nodes to **Redis Streams**.
- [ ] Stateless worker process: consume, run agent, write results via the Postgres
      store; consumer groups for at-least-once delivery.
- [ ] Demonstrate that idempotency (Phase 5) makes at-least-once safe.

**Acceptance:** the research workflow runs across ≥2 worker processes; killing one
worker mid-run does not corrupt state or duplicate side effects; a short write-up
explains what changed from v1 and, crucially, what did **not** (validation, ready-set,
policy) — that unchanged core is the architecture payoff.

---

## Cross-cutting, do continuously

- **Load test** (Locust/k6 or an async driver) once Phase 4 lands; record the
  bottleneck you found and how you addressed it — one paragraph in the README.
- **README** with the architecture diagram, the DR-1 single-vs-distributed writeup,
  and the load-test result. This is the highest-leverage prose in the whole project.
- Keep AGENTS.md §10, ARCHITECTURE.md, and this file in sync with reality as the code
  moves.
