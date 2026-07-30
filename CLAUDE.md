# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read the contract first

**[`docs/AGENTS.md`](docs/AGENTS.md) is the operating contract and it wins over anything
here.** It covers the golden rules, code conventions, testing conventions, boundaries, and
commit style. Read it, plus [`docs/SPEC.md`](docs/SPEC.md) (what),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (how and why — the decision records are the
part a reviewer weighs most), and [`docs/ROADMAP.md`](docs/ROADMAP.md) (the phased plan,
all eight phases now complete).

This file deliberately does **not** restate those rules. The repo forbids second sources of
truth — root `AGENTS.md` says so about itself — and a duplicated rule is a rule that drifts.
What is here is the orientation you cannot get by reading one file, and the traps that have
actually cost time.

## Commands

```bash
uv sync                                    # install into .venv
uv run pytest                              # full suite (~6s, no network)
uv run pytest tests/graph -q               # scope to one package
uv run pytest tests/store/test_postgres.py::test_the_dsn_variable_has_exactly_one_definition
uv run pytest -k "reclaim or idempotency"  # by keyword
uv run ruff check .                        # lint
uv run ruff format .                       # format — never hand-format
uv run mypy dagent                         # must be clean, no new ignores
uv run python benchmarks/load.py           # engine overhead vs the SPEC target
```

`pytest`, `ruff check`, and `mypy` must all pass before any task is complete.

### Running the tests that skip by default

83 tests skip without a database and a broker, and they are the ones covering the Postgres
store and the Redis transport. A green local run proves less than a green CI run:

```bash
docker run -d --rm --name dagent-pg -e POSTGRES_PASSWORD=dagent -e POSTGRES_DB=dagent \
  -p 5442:5432 postgres:16-alpine
docker run -d --rm --name dagent-redis -p 6399:6379 redis:7-alpine

DAGENT_TEST_POSTGRES_DSN="postgresql://postgres:dagent@localhost:5442/dagent" \
DAGENT_TEST_REDIS_URL="redis://localhost:6399/0" \
  uv run pytest                            # 1088 pass with both up, vs 1005/83 skipped
```

Set both if you touch `store/` or `transport/`. CI provisions both and fails the build if a
service-backed test was skipped.

### The CLI

```bash
uv run --env-file .env dagent run examples/research_dynamic.yaml --run-id r1
uv run dagent validate <file>              # submit-time checks, nothing executed
uv run dagent resume <run-id>              # needs --store postgres to be meaningful
uv run dagent inspect <run-id>             # full run reconstruction as JSON
uv run dagent worker --queue redis --store postgres    # a distributed worker
```

Credentials come from the environment only — run under `uv run --env-file .env`.

## Architecture: the parts that need several files to see

**One run loop, four configurations.** `Executor._loop` (`dagent/runtime/executor.py`) is
the only scheduler. A fresh run and a resumed one share it; so do a single-process run and a
distributed one. `run` seeds state with every node `PENDING`, `resume` seeds it from the
store, and the transport is swapped by injecting a different `Dispatcher`. If you find
yourself writing a second loop for any of those cases, that is the design error — the claim
of DR-4 and DR-12 is precisely that none of them is special.

**Two protocol seams, each with two implementations and one conformance suite.**

| Seam | Protocol | Impls | Contract test |
|---|---|---|---|
| Durability | `StateStore` (`store/base.py`) | `memory.py`, `postgres.py` | `tests/store/test_conformance.py`, `store` fixture |
| Transport | `WorkQueue` (`transport/base.py`) | `memory.py`, `redis.py` | `tests/transport/test_queue_conformance.py`, `queue` fixture |

Both fixtures are parametrised over every implementation. Add an impl → it is held to the
existing contract automatically. Add a behaviour → put it in the conformance file, not in
an impl-specific one, or the two will drift.

**The scheduler/unit-of-work split.** `executor.py` schedules a graph; `node.py`
(`NodeRunner`) runs one node under its policy — attempts, timeout, permits, budget,
recording, and the order the writes happen in. `dispatch.py` is the seam between them
(`LocalDispatcher` = an asyncio task, `QueueDispatcher` = a message). `worker.py` is
`NodeRunner` with a queue in front. Node-execution behaviour belongs in `node.py`, or the
distributed path silently gets a different policy layer than the local one.

**The graph has exactly one owner.** Expansion (a planner adding nodes at run time) is
validated against the whole graph, and `RunGraph.apply` contains **no `await`** — on one
event loop the absence of a suspension point *is* the serialisation, and
`tests/runtime/test_run_graph.py` asserts that by AST. Across processes the coordinator
owns it: a worker's request rides home on the result and is merged there. A node hands its
request to an `ExpansionSink`, and which sink it gets is the whole local/distributed
difference.

**The idempotency key `(run_id, node_id, attempt)` threads everything.** It is
`ctx.idempotency_key` to an agent, the entire payload of a work message, and the reason
at-least-once delivery is safe. An interrupted node is re-dispatched at the *same* attempt;
a retry after a definite failure gets a fresh one. Never renumber an attempt on resume or
redelivery.

**Layering is enforced by AST, not convention.** `tests/test_isolation.py` holds
`ALLOWED_DEPENDENCIES`: `models` → `graph`/`policy`/`store`/`transport` → `observability` →
`runtime`, with `agents` on top. It also bans model SDKs and HTTP outside `agents/`, bans
orchestration frameworks everywhere, and pins each optional dependency to the single module
allowed to import it (`asyncpg` → `store/postgres.py`, `redis` → `transport/redis.py`,
`opentelemetry.sdk` → `observability/setup.py`). A plain install must stay importable with
every extra absent.

**Determinism is structural.** Time comes from an injected `Clock`, retry jitter from an
injected function, model calls from an injected client. `ready_set` returns declaration
order, and completions are folded in declaration order — `asyncio.wait` returns a *set*, and
folding in set order once made a run's record depend on hash iteration. Timeouts are the one
deliberate exception (DR-10): only the event loop can cancel at a deadline.

## Traps that have actually cost time here

- **Test file basenames must be unique across `tests/`.** There are no `__init__.py` files,
  so `tests/a/test_x.py` and `tests/b/test_x.py` collide at collection. Hit twice; hence
  `test_run_graph.py` and `test_queue_conformance.py` are named as they are.
- **A passing test is not evidence until you have seen it fail.** Mutation-test anything
  load-bearing: break the behaviour, confirm the test catches it, restore. This has caught
  three tests here that could not fail — most recently an idempotency test where every
  attempt in the scenario was already `0`, so "uses the attempt it was given" and "always
  starts at zero" were indistinguishable.
- **Verify against the real thing before claiming a phase.** Postgres and Redis in Docker,
  the live provider through the CLI, and for Phase 8 actual `kill -9` on a worker process.
  The two nastiest bugs in the project — a lost expansion on resume, and a schema-creation
  race — were both invisible to a green suite.
- **A test that skips proves nothing.** Skips are deliberate and reasoned, never silent.
- **`--store memory` cannot outlive its process**, so `resume` and `inspect` against it are
  meaningless from a second command. Distributed runs require `postgres`.

## Current state

All eight roadmap phases are complete: 1005 tests offline / 1088 with services, 99%
coverage, ruff and mypy clean. Two things remain open and are documented rather than hidden:
CI has never actually run (there is no remote, so Phase 0's "CI green" criterion is formally
unmet), and the incremental ready-set frontier is deliberately deferred — the README names
the threshold, the measured cost, and the fix.
