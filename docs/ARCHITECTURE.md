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
Frozen pydantic v2 schemas — no logic. `Workflow`, `Node`, `Policy`, the state records
`NodeStateRecord`, `RunStateRecord`, and the model-I/O records `ModelRequest`,
`ModelResponse`, `ModelCallRecord`. Two enums that must never be conflated:
`NodeState {PENDING, READY, RUNNING, SUCCESS, FAILED, SKIPPED}` and
`RunState {PENDING, RUNNING, SUCCEEDED, FAILED, BUDGET_EXCEEDED, CANCELLED}`.

A node carries two kinds of data and the distinction matters: `inputs` is what other
nodes produced, `params` is what the author wrote down. `params` is what lets a workflow
have a source at all — every other node is fed from upstream, so something has to supply
the first value — and it is how a Phase 6 planner will hand each node it generates its
own subtopic. It is part of the frozen definition, so it is identical on every replay.

`NodeOutput` is `pydantic.JsonValue` rather than `Any`: FR-4 requires a *serializable*
output and Phase 5 has to write these to Postgres, so the constraint is cheaper enforced
at the type level today than discovered at the storage boundary later.

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
- `descendants(workflow, node_id)` → the blast radius of a failure: every node that can
  never become ready once `node_id` fails. Iterative, for the same reason cycle detection
  is. This is what `skip_downstream` marks.
- `expand_workflow(workflow, added, *, known_agents)` → the augmented workflow, or a
  rejection. **Append-only**, and that is the guardrail rather than an implementation
  detail — see §4. Restating an existing node *identically* is a no-op, which is what lets
  a planner be re-executed after a crash.
- `build_node(...)` → one node with the `depends_on` edges its inputs imply. Shared with
  `WorkflowBuilder`, so a planner-generated node is held to the standard a hand-written one
  is, rather than to a second one that drifts.
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
  `AgentContext` (resolved inputs, `params`, `run_id`/`node_id`/`attempt`, the injected
  `Clock`, and the model client). Context is how we keep agents pure and replayable.
  `ctx.idempotency_key` renders `(run_id, node_id, attempt)` as a string an agent can
  hand straight to whatever the outside world dedupes on — see §4.
- `clock.py` — the `Clock` `Protocol` (`now`, `sleep`) with `SystemClock` for production
  and `ManualClock` for tests. `SystemClock` is the only code in the package that touches
  real time.
- `model.py` — the `ModelClient` `Protocol` (one method, `complete`), plus
  `NullModelClient` (refuses loudly, so a run with no provider fails with a sentence
  rather than an `AttributeError`) and `StubModelClient` (scripted, for tests). Provider
  -agnostic and HTTP-free: a concrete provider lives in `agents/`.
- `metering.py` — `BudgetedModelClient`, which asks the run's `Budget` for permission
  before a call and charges it afterwards. Lives here rather than in `policy/` because it
  is the one piece of the budget story that has to know what a model call *is*; `Budget`
  itself holds numbers only, which is what keeps `policy` free of any dependency on the
  model seam. The executor stacks it *outside* the recorder, so a refused call is never
  recorded — there is no response, and a replay must not find a call that never happened.
- `recording.py` — `RecordingModelClient`, which wraps any client and writes every
  request/response into the run record keyed by `(run_id, node_id, attempt, sequence)`.
  That key is what a replay client will look calls up by. The wrapper is per-node and
  cheap; the client it wraps is shared for the whole run, which is how "one shared
  `httpx.AsyncClient`, never per-call" survives being wrapped.
- `registry.py` — name → agent *factory*, populated by a `@register("name")` decorator.
  Factories rather than instances, so each node gets a fresh agent and one node's state
  cannot leak into another's. Registries are ordinary objects the executor takes by
  injection; the module-level `default_registry` is a convenience nothing in the engine
  depends on. Re-registering a name raises rather than silently replacing — otherwise a
  workflow's behaviour would depend on import order.
- `expansion.py` — `Expansion`, the request an agent fills in via `ctx.expand(...)`, and
  `RunGraph`, the live definition for one run. `RunGraph.apply` **contains no `await`**,
  and that is load-bearing: on a single event loop the absence of a suspension point is
  what makes validate-and-merge atomic, so two planners finishing in the same batch are
  applied one after the other and the second is checked against a graph that already
  includes the first. No lock, no queue, and no window in which the graph is half-expanded.
  `tests/runtime/test_run_graph.py` asserts the absence of `await` by AST, because a stray
  one would reintroduce that window silently.
- `executor.py` — the scheduler. Owns the run loop, the ready-set recomputation, the
  concurrency permits, the handoff to the policy layer, and `resume`. A fresh run and a
  resumed one share one loop: `run` seeds it with every node `PENDING`, `resume` seeds it
  from the store. A resume that went down a second code path would be a second set of
  scheduling bugs, and the claim of DR-4 is precisely that resume is *not* special.
  This is the heart of the project; keep it under tight test. `run_id` is supplied by
  the caller rather than minted here, so no run depends on an id it cannot reproduce.

### Policy (`dagent/policy`)
Computes; never acts. This package decides *how long* to wait, *whether* a failure earns
another attempt, *who* may run, and *how much* may be spent — and the executor does the
waiting, dispatching, and calling. Its only dependencies are `dagent.errors` and
`dagent.models`, which is not a style preference: `dagent/runtime/__init__.py` imports the
executor, so a policy module importing anything under `dagent.runtime` would make the whole
package unimportable. `tests/test_isolation.py` enforces that arrow, and every other one
between the layers, by AST.

- `retry.py` — attempts, exponential backoff with full jitter, retryable-error
  classification. The jitter is an injected function (`full_jitter` in production,
  `no_jitter` in tests) rather than a call to `random`, for the same reason the clock is
  injected: a run has to be able to reproduce its own delays. **Full** jitter rather than a
  fixed exponential delay because identical backoff makes every node that failed against a
  rate-limited provider retry at the same instant, re-creating the overload.
- `limits.py` — a global `asyncio.Semaphore` and a dict of per-provider semaphores; a
  node acquires both before running (ordered acquisition to avoid permit deadlock).
  Permits are taken **per attempt**, not per node: holding a slot through a backoff sleep
  blocks a node that could be running. `Budget` tracks tokens/cost and refuses admission
  past the ceiling. It distinguishes *exceeded* (a ceiling has been reached) from
  *refused* (it actually stopped something), because a ceiling crossed by the last call of
  an otherwise complete run stopped nothing, and reporting that run as `BUDGET_EXCEEDED`
  would turn a success into a failure.
- `run.py` — `FailureMode` (`run_to_completion` / `fail_fast` / `skip_downstream`) and
  `RunPolicy`, the bundle the executor is handed. Node-level policy belongs to the frozen
  *definition*; run-level policy belongs to this *execution* and is supplied at submit
  time. Every default is inert — no cap, no ceiling, one attempt, no timeout — so a run
  that passes no policy behaves exactly as it did before the policy layer existed.

**No price table ships here.** Per-token prices are vendor knowledge that changes without
warning, and a stale table baked into an execution engine reports confident, wrong numbers.
The token ceiling is exact and needs no table; a run wanting a dollar ceiling injects the
pricing it trusts (`runtime/metering.py`'s `Pricer`).

### Store (`dagent/store`)
`StateStore` `Protocol`: `checkpoint`, `save_workflow`, `save_node_state`,
`append_output`, `append_model_call`, `load_run`, `load_workflow`, `load_output`,
`load_model_calls`. Two impls: `memory.py` (v1, dict-backed) and `postgres.py`
(`asyncpg`, behind the `postgres` extra). All durability semantics live behind this
protocol so the executor is storage-agnostic — a claim only worth making if both
implementations actually agree, which is what `tests/store/test_conformance.py` runs the
same contract against each of them to establish.

The **definition is persisted with the run** (`save_workflow`), which is what lets
`resume(run_id)` take no other argument. Rebuilding the graph from the original file
instead would work today and break in Phase 6, where a planner adds nodes at run time that
no file contains. It is a separate write rather than a field on `RunStateRecord` because
`checkpoint` replaces the whole record on every run-level transition, and a graph is a lot
of bytes to rewrite in order to say "still going".

Every write is safe to repeat. A resumed node re-writes the state and output it may
already have written, so the Postgres impl is entirely `INSERT ... ON CONFLICT DO UPDATE`
— Phase 8's at-least-once requirement arriving early, on purpose (DR-4).

`load_output` is what lets the executor resolve a node's inputs *through the store*
rather than from an in-process cache, keeping the store the single source of truth — and
turning Phase 5's resume into a reload rather than a reconstruction. A missing run or
output raises `StoreError` rather than returning `None`, because `None` is itself a legal
output and absence has to be signalled out of band.

Three ordering guarantees the executor makes and an implementation may rely on: the
definition is written before the run record, so a run that exists can always be resumed;
run-level state is written before the first node starts; and a node's output is written
*before* that node is marked `SUCCESS`, so a node found `SUCCESS` on reload always has an
output to read.

That first rule has a visible consequence in the Postgres schema: `dagent_workflow` is the
one table with no foreign key back to `dagent_run`, because a foreign key would demand the
parent row first and invert the guarantee. Of the two torn writes available — a definition
with no run, or a run with no definition — the first is an orphan row and the second is a
run nobody can continue.

### Agents (`dagent/agents`)
Concrete implementations: `planner`, `researcher`, `synthesizer`, `critic` — plus the
provider clients they talk through, and the no-I/O fakes (`constant`, `echo`, `fake`).
These are the *only* modules allowed to import a model SDK, or to know a vendor's name at
all. Each is small and independently testable with a fake model client.

`gemini.py` is the v1 provider, written against the REST endpoint over `httpx` rather
than a vendor SDK: the surface it needs is one POST, so going direct keeps the dependency
list honest and the wire format visible in the code instead of behind a client library.
Keys come from the environment and nowhere else.

Importing the package registers its agents on `default_registry` under the names a
workflow file uses. `FailingAgent` is deliberately *not* registered — a workflow should
never be able to name it by accident.

### Loading (`dagent/loader.py`)
FR-1 wants workflows definable in Python *and* in a file. `load_workflow` parses YAML and
builds through `WorkflowBuilder`, so files get no separate rulebook — the same validator
runs on both paths. It lives at the package root rather than in `graph/` because reading
a file is I/O and `graph/` is proven pure by `tests/graph/test_purity.py`; parsing
(`load_workflow`, pure, takes text) is split from reading (`load_workflow_file`) for the
same reason.

**Stack note (AGENTS.md §4):** this adds `pyyaml` to the dependency list. It is the
minimum needed to satisfy FR-1's "loadable from YAML", it is used only through
`safe_load` so a workflow file can never construct arbitrary Python, and it is confined
to this one module.

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
4. Terminate when nothing is in flight *and* the ready set is empty. Derive `RunState`
   from the node states + budget outcome.

Step 4 is deliberately "no more progress is possible", not "no node is still `PENDING`".
A node whose dependency failed never becomes ready, so it stays `PENDING` forever and a
loop waiting for `PENDING` to empty would never exit.

What happens to that node is a *policy* decision, not a scheduler one, so it is one of
three configurable failure modes:

| Mode | In-flight siblings | Blocked dependents | Further dispatch |
|---|---|---|---|
| `run_to_completion` (default) | finish | left `PENDING` | continues |
| `skip_downstream` | finish | marked `SKIPPED` | continues |
| `fail_fast` | cancelled, recorded `FAILED` | left `PENDING` | stops |

`SKIPPED` and cancelled-`FAILED` are kept distinct on purpose: `SKIPPED` means *never
dispatched, and never can be*, while a node cancelled by `fail_fast` did start and did not
finish. Conflating them would lose the difference between "we chose not to" and "we
stopped it".

Fan-out is step 2 dispatching multiple ready nodes at once; fan-in is step 1
re-detecting a node whose several deps just completed.

**One subtlety worth stating:** `asyncio.wait` returns a *set*, and several nodes can land
in one batch. Folding them in set order would make the record depend on hash iteration —
two identical runs could attribute the same skipped node to different upstream failures.
Completions are therefore folded in declaration order, the same tiebreak `ready_set` uses.

## 4. The hard problems (this is the senior signal)

### Durable, resumable execution
State transitions are written **before** side effects where possible, and each node run
carries an idempotency key `(run_id, node_id, attempt)`, surfaced to agents as
`ctx.idempotency_key`. On resume, the executor loads persisted state and:

| Node was left | Resume does |
|---|---|
| `SUCCESS` | skips it — its output is already in the store for dependents to read |
| `RUNNING` / `READY` | re-dispatches it **at the same attempt number** |
| `FAILED` / `SKIPPED` | leaves it; a verdict was reached |
| `PENDING` | picks it up normally |

The same-attempt rule is the whole design. A crash cannot tell you whether the side effect
landed before the process died, so the node comes back under the key the outside world has
already seen, and an agent that deduplicates on that key commits exactly once. A retry
after a *definite* failure is a different matter and gets a fresh attempt, because that
work provably did not complete.

Budget usage is rebuilt from the recorded model calls before anything re-runs — a per-run
ceiling that reset itself on every crash would be a ceiling per crash.

The design principle: **make re-execution safe first, then make resume trivial.** This is
deliberately the Temporal problem in miniature, and the README should say so.

### Dynamic DAG expansion
An agent calls `ctx.expand(...)` with nodes built by `build_node`. The request is collected
on its context and applied **only once that attempt succeeds** — an attempt that expands and
then fails leaves nothing behind, so the retry starts from a graph nobody has already
grown. The executor then: (1) revalidates the *augmented* graph, (2) persists it, (3)
records the new nodes as `PENDING`, and (4) lets the normal ready-set loop pick them up. A
node absent from the state map counts as `PENDING`, which Phase 1 chose for exactly this
moment: expansion needs no state backfill and no second scheduling path.

**The invariant that prevents deadlock is that expansion is append-only.** Existing nodes
are carried over untouched and can never gain a dependency, because a node already
`RUNNING` or `SUCCESS` re-checks nothing and would be stranded by one. The *reverse* is
always safe: a new node may depend on an existing node in any state, including one that has
already finished — which is precisely how a planner's generated children depend on the
planner that generated them, and how the showcase workflow works at all.

That also decides where the fan-in lives. The synthesizer cannot be written into the file,
because it has to depend on nodes that did not exist when the file was written; the planner
emits it alongside the researchers it fans in.

Two bounds, for two different failure modes. `max_expansion_depth` (default 1: a planner
may plan, and what it plans may not plan further) stops an agent that emits a copy of
itself. `max_graph_nodes` stops any runaway however it arose — and unlike depth, which is
counted in memory and starts fresh on resume, it is measured against the persisted graph,
so it is the bound that survives a crash.

A rejected expansion fails *that node* and nothing else: `ValidationError` reaches the
executor as an ordinary node failure, the default retry classification does not retry it,
and the run carries on and terminates normally. A bad planner cannot deadlock a run.

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

**DR-7: Errors are classified retryable at the point they are raised.**
FR-5 says only retryable errors are retried, and the code best placed to judge is the code
that failed: a provider client knows a 429 is weather and a 400 is a bug in the request.
So `AgentError` carries a `retryable` flag, defaulting to `True` because most agent
failures are transient provider trouble, and the classifier the policy layer applies is
itself injectable. The default rule is deliberately narrow — an `AgentError` that has not
marked itself permanent, or a `TimeoutError`. A `ValidationError` is as true next time; a
`StoreError` means the thing recording the attempt is broken; a `TypeError` is a bug, and
retrying it buys three identical stack traces and three times the latency. *Rejected:*
retrying every exception (turns bugs into slow bugs), or a central table of retryable
types (puts vendor knowledge in the core and goes stale silently).

**DR-8: A planner is an ordinary agent that happens to return a graph-shaped opinion.**
Expansion is requested through `ctx.expand(...)` on the context every agent already has,
rather than through a second `Planner` protocol or a magic key in the returned output. One
agent contract means the registry, the retry loop, the timeout, the budget, the recording
wrapper and the idempotency key all apply to a planner unchanged — none of them had to
learn what a planner is. It also means a planner can both expand *and* produce an ordinary
output, which is how the showcase reports the plan it chose. *Rejected:* a separate
protocol (two contracts to keep in step, and an awkward answer to "what is the node's
output?"), and a sentinel key in the output (stringly-typed, and indistinguishable from an
agent that happens to return that key).

**DR-9: The workflow definition is persisted with the run, and `resume` takes only a
run id.** The alternative — hand `resume` the file again — is what Airflow and Temporal do,
and it is defensible: the definition is code, the state is data. It breaks the moment a
planner grows the graph at run time (Phase 6, FR-7), because the augmented graph exists in
no file. Storing it also removes a whole class of bug where a run silently continues
against an edited definition. *Rejected:* reconstructing from the file (unresumable once
expansion lands), and embedding the graph in `RunStateRecord` (rewritten on every
checkpoint, and it would make `models/state.py` import `models/workflow.py` in a cycle).

**DR-10: Timeouts are measured by the event loop, not by the injected `Clock`.**
Every other read of time in dagent goes through the `Clock` seam, including retry backoff
— which is why a test can assert a run waited four seconds without waiting. Timeouts are
the deliberate exception: cancelling a coroutine at a deadline is something only the loop's
own scheduler can do, and a wall-clock seam cannot make an `await` return. Backoff and
timeout therefore sit on different clocks by design: backoff is a *duration the engine
chooses* and must replay identically, while a timeout is a *deadline imposed on work whose
duration the engine does not control*. Also why the timeout starts *inside* the concurrency
permits rather than outside them — queueing for a slot is contention, not the node's own
latency, and a node that never got to run has not overrun anything.
