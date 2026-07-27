# ARCHITECTURE.md — Dagent

How the system is built and *why* it's built that way. Read `SPEC.md` first for the
requirements this satisfies. The decision records (§7) are the part a reviewer will
weigh most — they show judgment, not just code.

## 1. Shape of the system

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

Everything the executor does is a loop: compute ready set → acquire permits →
dispatch → on completion persist state and recompute ready set → repeat until the
graph is terminal.

## 2. Components

### Models (`dagent/models`)
Frozen pydantic v2 schemas — no logic. `Workflow`, `Node`, `Policy`, and the state
records `NodeStateRecord`, `RunStateRecord`. Two enums that must never be conflated:
`NodeState {PENDING, READY, RUNNING, SUCCESS, FAILED, SKIPPED}` and
`RunState {PENDING, RUNNING, SUCCEEDED, FAILED, BUDGET_EXCEEDED, CANCELLED}`.

### Graph (`dagent/graph`)
Pure functions over a `Workflow`.
- `validate(workflow, *, known_agents=None)` → raises `ValidationError` on cycle /
  unsatisfied input / unknown agent, else returns nothing. Cycle detection via DFS
  coloring (iterative, so a deep chain can't exhaust the stack); the reported error
  includes the cycle path both in its message and on the exception's `.cycle`.
  The registry is **injected, not imported**: `graph` must not depend on `runtime`, so
  the caller supplies the registered agent names and omitting them skips that one check.
  This keeps validation usable — and testable — with no registry in existence.
- `ready_set(workflow, states)` → nodes whose deps are all `SUCCESS` and which are still
  `PENDING`, returned as a **tuple in declaration order** rather than an unordered set,
  so dispatch order reproduces on replay. A node absent from `states` counts as
  `PENDING`, so an expanded graph needs no state backfill. Called after every completion;
  keep it O(edges).
- `topological_order(workflow)` → all node ids in dependency order (Kahn, declaration
  order as the tiebreak).
- `WorkflowBuilder` → the typed builder. Ergonomic, not lenient: it adds each input's
  source node to that node's `depends_on` and then runs the same `validate`.
- **An input may only read from an ancestor.** Reading the output of a node you don't
  wait for is a race, not a dependency, so `validate` rejects it; that is what FR-2's
  "unsatisfied input" means here. `depends_on` remains meaningful on its own for
  ordering-only edges — waiting for a node whose output you don't read.
No async, no I/O — trivially unit- and property-testable, and enforced by
`tests/graph/test_purity.py`.

### Runtime (`dagent/runtime`)
- `agent.py` — the `Agent` `Protocol` (`async def run(ctx) -> Output`) and
  `AgentContext` (resolved inputs, `run_id`/`node_id`, injected `Clock`, model client
  factory, `Budget` handle). Context is how we keep agents pure and replayable.
- `registry.py` — name → agent factory, populated by a `@register("name")` decorator.
- `executor.py` — the scheduler. Owns the run loop, the ready-set recomputation, the
  concurrency permits, and the handoff to the policy layer. This is the heart of the
  project; keep it under tight test.

### Policy (`dagent/policy`)
- `retry.py` — attempts, exponential backoff with full jitter, retryable-error
  classification.
- `limits.py` — a global `asyncio.Semaphore` and a dict of per-provider semaphores; a
  node acquires both before running (ordered acquisition to avoid permit deadlock).
  `Budget` tracks tokens/cost and refuses admission past the ceiling.

### Store (`dagent/store`)
`StateStore` `Protocol`: `save_node_state`, `load_run`, `append_output`,
`checkpoint`. Two impls: `memory.py` (v1, dict-backed) and `postgres.py` (v2,
`asyncpg`). All durability semantics live behind this protocol so the executor is
storage-agnostic.

### Agents (`dagent/agents`)
Concrete implementations: `planner`, `researcher`, `synthesizer`, `critic`. These are
the *only* modules allowed to import a model SDK. Each is small and independently
testable with a fake model client.

### Observability (`dagent/observability`)
OpenTelemetry setup, metric definitions, and `structlog` configuration. Executor and
policy emit spans/metrics through here; nothing here knows about specific agents.

## 3. Execution model

Single-process `asyncio`. The executor holds:
- the frozen `Workflow` (mutable only via validated expansion),
- an in-memory view of `NodeState` kept consistent with the `StateStore`,
- a set of in-flight `asyncio.Task`s.

Loop:
1. `ready = ready_set(workflow, states)`.
2. For each ready node: mark `READY`, persist, spawn a task that (a) acquires global +
   provider permits, (b) runs the node under the policy layer, (c) on return persists
   `SUCCESS`/`FAILED` and the output.
3. `await` the first task to finish (`asyncio.wait(FIRST_COMPLETED)`), fold its result
   into `states`, then go to 1.
4. Terminate when no nodes are `PENDING`/`READY`/`RUNNING`. Derive `RunState` from the
   node states + budget outcome.

Fan-out is step 2 dispatching multiple ready nodes at once; fan-in is step 1
re-detecting a node whose several deps just completed.

## 4. The hard problems (this is the senior signal)

### Durable, resumable execution
State transitions are written **before** side effects where possible, and each node
run carries an idempotency key `(run_id, node_id, attempt)`. On resume, the executor
loads persisted state and: any node already `SUCCESS` is skipped; any node `RUNNING`
at crash time is re-dispatched, and because the agent's side effect is guarded by the
idempotency key, re-execution does not double-commit. The design principle:
**make re-execution safe first, then make resume trivial.** This is deliberately the
Temporal problem in miniature, and the README should say so.

### Dynamic DAG expansion
A planner returns new node definitions. The executor: (1) revalidates the *augmented*
graph (must stay acyclic), (2) inserts nodes as `PENDING`, (3) lets the normal
ready-set loop pick them up. The invariant that prevents deadlock: expansion only ever
*adds* nodes and edges among new/incomplete nodes — it never adds a dependency onto an
already-`SUCCESS` node in a way that could strand a running branch. Expansion is
serialized through the executor's single loop, so there's no concurrent mutation of the
graph.

### Backpressure & limits
Model APIs are rate-limited and cost money, so concurrency is gated on two axes at
once (global + provider) and admission is gated on budget. Because a node needs
multiple permits, they're always acquired in a fixed order to avoid a permit deadlock,
and released in `finally`. The ready set can be large while in-flight stays capped —
that gap *is* the backpressure.

### Deterministic replay
The `AgentContext` injects the clock and the model client, and a recording wrapper
logs every model request/response and clock read into the run's record. Replay swaps
the live client for one that serves recorded responses, so a run reproduces exactly
with no network. This makes crash-resume testable and debugging offline-able.

## 5. Data flow for one node

`ready_set` picks node → executor resolves `inputs` from upstream outputs in the store
→ builds `AgentContext` → policy wraps the call (timeout + retry + budget admission) →
agent `run(ctx)` executes → output serialized and persisted → state → `SUCCESS` →
ready set recomputed. A failure instead flows through retry; exhausted retries set
`FAILED` and the run's failure semantics decide siblings/downstream.

## 6. Scaling path (v1 → v2)

v1 is one process. To distribute: replace the in-process task dispatch with a
**Redis Streams** work queue — the executor becomes a *coordinator* that enqueues
ready nodes; stateless **workers** consume, run the agent, and write results back
through the `StateStore` (now Postgres). The ready-set/validation/policy logic is
unchanged because it was never coupled to the transport. Consumer groups give
at-least-once delivery, which is safe precisely because node execution is already
idempotent. This is the whole reason v1 invests in idempotency early.

## 7. Decision records

**DR-1: Single-process asyncio for v1, distributed workers for v2.**
Chosen because the valuable, hard-to-get-right parts (ready-set scheduling, dual
concurrency gating, durable idempotent resume) are fully expressible in one process
and easier to test there. Distribution is added by swapping the transport, not
redesigning the core. *Rejected:* starting distributed — it would couple correctness
work to infrastructure and slow the parts that actually demonstrate skill.

**DR-2: Custom engine, no orchestration framework.**
The entire point is demonstrating that the execution engine was built, not glued.
A provider SDK inside an agent is fine; an orchestration framework anywhere defeats
the exercise. *Rejected:* LangGraph/Airflow/Temporal-as-dependency.

**DR-3: `StateStore` protocol with in-memory + Postgres impls.**
Keeps the executor storage-agnostic and lets v1 run with zero external services while
v2 gains durability by swapping the impl. *Rejected:* hard-coding Postgres (slows the
inner test loop, forces infra on day one).

**DR-4: Idempotency before durability.**
Resume is easy *if* re-execution is safe; unsafe re-execution makes resume a source of
double side effects. So the idempotency key lands with node execution, before any
resume feature. *Rejected:* bolting on resume first and "handling" duplicates later.

**DR-5: Dependency injection for all non-determinism.**
Clock and model client are injected via `AgentContext`, enabling deterministic replay
and network-free tests. *Rejected:* calling `time`/model SDKs directly in core logic —
it makes the system untestable and unrepayable.

**DR-6: Dual-axis concurrency (global + per-provider) via ordered semaphore
acquisition.** Reflects the real constraint (providers rate-limit independently) and
prevents one provider's slowness from starving others past the global cap. Fixed
acquisition order avoids permit deadlock. *Rejected:* a single global cap (ignores
per-provider limits) or per-provider only (can blow past total resource budget).
