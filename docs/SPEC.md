# SPEC.md — Dagent

The functional and non-functional specification. `ARCHITECTURE.md` explains *how*
these are realized; `ROADMAP.md` sequences the build.

## 1. Purpose

Dagent executes multi-agent AI workflows defined as a directed acyclic graph, where
each node is an agent or tool. It runs independent nodes concurrently, resolves data
dependencies, enforces retry/timeout/budget policy, and persists state so a crashed
run resumes instead of restarting. The engine is model-agnostic and framework-free.

Success for this project = a reviewer reads the code and thinks *"this person can
build a durable distributed execution system,"* not *"this person can call an LLM."*

## 2. Core concepts

- **Workflow** — an immutable definition: a set of nodes plus their data
  dependencies. Validated once at submission.
- **Node** — a named unit of work bound to a registered agent, with declared inputs
  (which upstream outputs it consumes) and a policy override (optional).
- **Agent** — the pluggable behavior behind a node. Implements a single async
  contract. May be an LLM call, a tool, or pure logic.
- **Run** — one execution of a workflow. Has a `RunState` and a per-node `NodeState`.
- **Context** — what an agent receives at execution: its resolved inputs, run
  metadata, the injected clock, the model client, and the budget handle.

## 3. Functional requirements

### FR-1 Workflow definition
- Workflows are defined in Python (typed builder) **and** loadable from YAML.
- A node declares: `id`, `agent` (registry name), `depends_on` (list of node ids),
  `inputs` (mapping of local name → upstream node output), and optional `policy`.
- Definitions are frozen after validation; runtime never mutates them.

### FR-2 Validation (at submit time, before any execution)
- Reject graphs containing a cycle, with the offending cycle reported.
- Reject a node whose declared input references an output no upstream node produces.
- Reject references to unregistered agents.
- Validation is pure and fast; a definition either fully validates or is rejected.

### FR-3 Scheduling & execution
- Compute the **ready set** = nodes whose dependencies are all `SUCCESS`.
- Execute ready nodes **concurrently** via `asyncio`.
- Respect a **global concurrency cap** and **per-provider concurrency caps**
  simultaneously (a node needs both permits to run).
- On node completion, recompute the ready set (fan-in) and dispatch newly-ready nodes.
- Pass upstream outputs to downstream nodes per their `inputs` mapping.

### FR-4 Agent contract
- Every agent implements `async def run(ctx: AgentContext) -> Output`.
- Agents register by name via decorator/entry point; adding one requires no change
  to the core.
- Agents receive resolved inputs and produce a serializable output.

### FR-5 Policy
- **Retry:** per-node max attempts with exponential backoff + jitter; only retry
  errors classified retryable.
- **Timeout:** per-node wall-clock timeout; a timed-out node is cancelled cleanly.
- **Failure semantics** (configurable per run): `fail_fast` (cancel siblings on first
  failure) vs `run_to_completion` (let independent branches finish) vs `skip_downstream`
  (mark dependents `SKIPPED`).
- **Budgets:** per-run token and cost ceiling; when exceeded, no new model calls are
  admitted and the run ends in a `BUDGET_EXCEEDED` terminal state.

### FR-6 Durability & resume
- Persist `NodeState` transitions and each node's output through the `StateStore`.
- After a process crash, a run reloads its state and continues from the last
  consistent checkpoint.
- Node execution is **idempotent**: re-executing a node after a crash must not
  double-commit side effects (dedupe via a per-(run, node, attempt) idempotency key).

### FR-7 Dynamic expansion
- A planner-type agent may emit new nodes at runtime (e.g. "research these 5
  subtopics"), expanding the graph mid-run.
- Newly added nodes are validated against the live graph (no cycles introduced) and
  scheduled by the same ready-set logic. Expansion must not deadlock or starve.

### FR-8 Deterministic replay
- Record all model I/O and other non-deterministic inputs per run.
- A recorded run can be replayed offline with no network access, reproducing the
  same node outputs — for debugging and for tests.

### FR-9 Observability
- One OpenTelemetry span per node, nested under a run span; spans carry attempt
  count, provider, token usage.
- Metrics: nodes in-flight, ready-set size, per-provider concurrency, retries,
  run duration, tokens/cost per run.
- Structured, correlated logs keyed by `run_id` and `node_id`.

### FR-10 Interfaces
- **CLI:** submit a workflow file, watch a run, inspect/resume a run by id.
- **Run inspector:** dump a run's node states, timings, and outputs (JSON) — enough
  to render a simple DAG view later.

## 4. Non-functional requirements

- **Correctness under concurrency** is the top priority: caps are never exceeded,
  no lost updates to run state, no deadlock on fan-in or expansion.
- **Resumability:** killing the process at any point and restarting yields the same
  final result as an uninterrupted run (verified by test).
- **Isolation:** core packages (`graph`, `runtime/executor`, `policy`, `store`)
  import no model SDK.
- **Performance target (v1):** overhead of the engine itself (excluding model
  latency) under a few milliseconds per node at 100 concurrent nodes on a laptop.
- **Extensibility:** a new agent or a new `StateStore` backend requires no change to
  the executor.
- **Portability:** v1 runs single-process with zero external services (in-memory
  store); Postgres and distributed workers are additive, not assumed.

## 5. Explicit non-goals (v1)

- No web UI (JSON inspector output is enough; a viewer can come later).
- No multi-tenant auth, no hosted API surface.
- No distributed execution in v1 — single-process asyncio. Distribution is the v2
  scaling story and must not complicate the v1 design.
- No general orchestration-framework dependency, ever.

## 6. Example workflow (the showcase)

A research assistant that exercises every hard feature:

1. **planner** decomposes a question into N subtopics (dynamic expansion).
2. **researcher** ×N investigate subtopics in parallel (fan-out, per-provider caps).
3. **synthesizer** merges researcher outputs (fan-in).
4. **critic** reviews; may loop back for one more synthesis pass (bounded conditional
   cycle, expressed as a re-plan rather than a static cycle).

This single workflow demonstrates dependency resolution, concurrency, dynamic graph
growth, policy, and durability — build and test against it end to end.
