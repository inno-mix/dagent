"""A Postgres-backed ``StateStore`` — the same protocol, state that outlives the process.

DR-3's payoff. The executor does not change by a line to use this: it was written against
:class:`~dagent.store.base.StateStore` and has never known which implementation it holds.
That is the whole point of the protocol, and this module is the proof.

Not imported by ``dagent.store``'s package init, and ``asyncpg`` is an optional extra
(``pip install dagent[postgres]``). SPEC's portability requirement says v1 runs with zero
external services; a store nobody asked for should not be able to break the import of a
run that stays in memory.

Everything is written through ``INSERT ... ON CONFLICT DO UPDATE``. The store is the
target of at-least-once writes even in v1 — a resumed node re-writes the state and output
it may already have written — so every write has to be safe to repeat. That is Phase 8's
requirement arriving early, on purpose (DR-4).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import asyncpg

from dagent.errors import StoreError
from dagent.models.model_call import ModelCallRecord, ModelRequest, ModelResponse
from dagent.models.state import NodeOutput, NodeState, NodeStateRecord, RunState, RunStateRecord
from dagent.models.workflow import Workflow
from dagent.store import POSTGRES_DSN_ENV

__all__ = ["PostgresStateStore"]

_SCHEMA_LOCK = 0x6461_6765  # "dage"
"""The advisory-lock key schema creation serializes on.

An arbitrary constant, and it only has to be arbitrary consistently: every dagent process
uses this one, and nothing else in the database is expected to. Advisory locks live in their
own namespace and are released when the transaction ends, so a process killed mid-DDL
releases it rather than wedging the next one.
"""

DSN_ENV = POSTGRES_DSN_ENV
"""Re-exported so provider-side code can name it without reaching into the package."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS dagent_run (
    run_id        TEXT PRIMARY KEY,
    workflow_name TEXT        NOT NULL,
    state         TEXT        NOT NULL,
    created_at    TIMESTAMPTZ,
    updated_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS dagent_workflow (
    run_id     TEXT PRIMARY KEY,
    definition JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS dagent_node_state (
    run_id      TEXT        NOT NULL REFERENCES dagent_run(run_id) ON DELETE CASCADE,
    node_id     TEXT        NOT NULL,
    state       TEXT        NOT NULL,
    attempt     INTEGER     NOT NULL DEFAULT 0,
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    error       TEXT,
    PRIMARY KEY (run_id, node_id)
);

CREATE TABLE IF NOT EXISTS dagent_output (
    run_id  TEXT  NOT NULL REFERENCES dagent_run(run_id) ON DELETE CASCADE,
    node_id TEXT  NOT NULL,
    output  JSONB NOT NULL,
    PRIMARY KEY (run_id, node_id)
);

CREATE TABLE IF NOT EXISTS dagent_model_call (
    run_id      TEXT        NOT NULL REFERENCES dagent_run(run_id) ON DELETE CASCADE,
    node_id     TEXT        NOT NULL,
    attempt     INTEGER     NOT NULL,
    sequence    INTEGER     NOT NULL,
    request     JSONB       NOT NULL,
    response    JSONB       NOT NULL,
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    PRIMARY KEY (run_id, node_id, attempt, sequence)
);
"""
"""The whole schema.

``dagent_workflow`` is a separate table rather than a column on ``dagent_run`` because
``checkpoint`` rewrites the run row on every run-level transition, and a graph is a lot of
bytes to rewrite in order to say "still going". The definition is written once.

It is also the one table with **no foreign key back to ``dagent_run``**, and that is
deliberate rather than an oversight. The store's ordering rule is that the definition is
written *before* the run record, so that a run which exists can always be resumed; a
foreign key would invert that and demand the parent first. Of the two possible torn
writes — a definition with no run, or a run with no definition — the first is an orphan
row and the second is a run nobody can continue. The invariant is worth more than the
referential tidiness, so the constraint is the thing that goes.

``dagent_model_call`` is keyed by ``(run_id, node_id, attempt, sequence)`` — the same key
the recording client uses — so a re-executed attempt overwrites its own calls rather than
appending a second copy a replay would then see twice.
"""


class PostgresStateStore:
    """Run state and outputs in Postgres, behind the ``StateStore`` protocol."""

    def __init__(self, pool: asyncpg.Pool[Any]) -> None:
        """Wrap an existing connection pool.

        A pool rather than a connection: the executor writes from many concurrent node
        tasks at once, and a single connection would serialise them into a queue behind
        the scheduler it is meant to be keeping up with.
        """
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str, **pool_options: Any) -> Self:
        """Open a pool and make sure the schema exists.

        Raises:
            StoreError: If the database is unreachable or refuses the schema.
        """
        try:
            pool = await asyncpg.create_pool(dsn, **pool_options)
        except (OSError, asyncpg.PostgresError) as exc:
            raise StoreError(f"cannot connect to Postgres: {type(exc).__name__}: {exc}") from exc
        if pool is None:  # pragma: no cover — asyncpg only returns None in a nested loop
            raise StoreError("asyncpg returned no connection pool")

        store = cls(pool)
        await store.create_schema()
        return store

    async def create_schema(self) -> None:
        """Create the tables if they are not already there.

        Idempotent, and deliberately not a migration framework: one `CREATE TABLE IF NOT
        EXISTS` per table is the right amount of machinery for a schema that is five tables
        and has never been versioned. When it needs versioning it will need Alembic, and
        that is a different change.

        Serialized by an advisory lock, because ``IF NOT EXISTS`` is not the concurrency
        guarantee it looks like: two connections that both find a table missing will both
        try to create it, and the loser gets a unique-violation on ``pg_type`` rather than a
        polite no-op. That could not happen while exactly one process ever connected, and it
        happens immediately in Phase 8, where every worker opens the store on startup and a
        pool comes up all at once. An advisory lock rather than catching the violation: the
        error arrives from Postgres' own catalogue and matching on it would be guessing at
        which of several catalogue constraints tripped.
        """
        async with self._acquire() as connection, connection.transaction():
            # Held until the transaction ends, which is what makes the check-then-create
            # below atomic against another connection doing the same thing.
            await connection.execute("SELECT pg_advisory_xact_lock($1)", _SCHEMA_LOCK)
            await connection.execute(SCHEMA)

    async def close(self) -> None:
        """Close the pool."""
        await self._pool.close()

    async def __aenter__(self) -> Self:
        """Enter a context that closes the pool on the way out."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the pool."""
        await self.close()

    # --- writes -----------------------------------------------------------------------

    async def checkpoint(self, run: RunStateRecord) -> None:
        """Write run-level state, replacing the stored record entirely.

        The node map travels with the record, so this writes the run row and every node
        row in one transaction — a half-applied checkpoint is a run whose summary
        disagrees with its own nodes.
        """
        async with self._acquire() as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO dagent_run (run_id, workflow_name, state, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (run_id) DO UPDATE
                SET workflow_name = EXCLUDED.workflow_name,
                    state         = EXCLUDED.state,
                    created_at    = EXCLUDED.created_at,
                    updated_at    = EXCLUDED.updated_at
                """,
                run.run_id,
                run.workflow_name,
                run.state.value,
                run.created_at,
                run.updated_at,
            )
            for record in run.nodes.values():
                await self._write_node_state(connection, record)

    async def save_workflow(self, run_id: str, workflow: Workflow) -> None:
        """Write the definition this run is executing."""
        async with self._acquire() as connection:
            await connection.execute(
                """
                INSERT INTO dagent_workflow (run_id, definition) VALUES ($1, $2)
                ON CONFLICT (run_id) DO UPDATE SET definition = EXCLUDED.definition
                """,
                run_id,
                workflow.model_dump_json(),
            )

    async def save_node_state(self, record: NodeStateRecord) -> None:
        """Write one node's state, replacing any previous state for that node."""
        async with self._acquire() as connection:
            await self._write_node_state(connection, record)

    async def append_output(self, run_id: str, node_id: str, output: NodeOutput) -> None:
        """Record the output produced by a node's latest attempt.

        Stored as JSON text in a ``JSONB NOT NULL`` column, so a node whose output *is*
        ``None`` stores the JSON literal ``null`` and is still distinguishable from a node
        that produced nothing at all: absence is a missing row, not a NULL.
        """
        async with self._acquire() as connection:
            await connection.execute(
                """
                INSERT INTO dagent_output (run_id, node_id, output) VALUES ($1, $2, $3)
                ON CONFLICT (run_id, node_id) DO UPDATE SET output = EXCLUDED.output
                """,
                run_id,
                node_id,
                json.dumps(output),
            )

    async def append_model_call(self, record: ModelCallRecord) -> None:
        """Record one model call made during this run."""
        async with self._acquire() as connection:
            await connection.execute(
                """
                INSERT INTO dagent_model_call
                    (run_id, node_id, attempt, sequence, request, response,
                     started_at, finished_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (run_id, node_id, attempt, sequence) DO UPDATE
                SET request     = EXCLUDED.request,
                    response    = EXCLUDED.response,
                    started_at  = EXCLUDED.started_at,
                    finished_at = EXCLUDED.finished_at
                """,
                record.run_id,
                record.node_id,
                record.attempt,
                record.sequence,
                record.request.model_dump_json(),
                record.response.model_dump_json(),
                record.started_at,
                record.finished_at,
            )

    # --- reads ------------------------------------------------------------------------

    async def load_run(self, run_id: str) -> RunStateRecord:
        """Return a run's state, including every node's state."""
        async with self._acquire() as connection:
            row = await connection.fetchrow(
                "SELECT workflow_name, state, created_at, updated_at "
                "FROM dagent_run WHERE run_id = $1",
                run_id,
            )
            if row is None:
                raise StoreError(f"unknown run {run_id!r}")
            nodes = await connection.fetch(
                "SELECT node_id, state, attempt, started_at, finished_at, error "
                "FROM dagent_node_state WHERE run_id = $1 ORDER BY node_id",
                run_id,
            )

        return RunStateRecord(
            run_id=run_id,
            workflow_name=row["workflow_name"],
            state=RunState(row["state"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            nodes={
                node["node_id"]: NodeStateRecord(
                    run_id=run_id,
                    node_id=node["node_id"],
                    state=NodeState(node["state"]),
                    attempt=node["attempt"],
                    started_at=node["started_at"],
                    finished_at=node["finished_at"],
                    error=node["error"],
                )
                for node in nodes
            },
        )

    async def load_workflow(self, run_id: str) -> Workflow:
        """Return the definition a run is executing."""
        async with self._acquire() as connection:
            definition = await connection.fetchval(
                "SELECT definition FROM dagent_workflow WHERE run_id = $1", run_id
            )
        if definition is None:
            raise StoreError(f"run {run_id!r} has no stored workflow")
        return Workflow.model_validate_json(definition)

    async def load_output(self, run_id: str, node_id: str) -> NodeOutput:
        """Return the output a node produced."""
        async with self._acquire() as connection:
            row = await connection.fetchrow(
                "SELECT output FROM dagent_output WHERE run_id = $1 AND node_id = $2",
                run_id,
                node_id,
            )
        if row is None:
            raise StoreError(f"run {run_id!r} has no output for node {node_id!r}")
        # `row["output"]` is the JSON text asyncpg read back from JSONB; `None` would mean
        # the row was missing, which the check above has already ruled out.
        loaded: NodeOutput = json.loads(row["output"])
        return loaded

    async def load_model_calls(self, run_id: str) -> Sequence[ModelCallRecord]:
        """Return every model call made during a run, ordered by its key."""
        async with self._acquire() as connection:
            if not await connection.fetchval("SELECT 1 FROM dagent_run WHERE run_id = $1", run_id):
                raise StoreError(f"unknown run {run_id!r}")
            rows = await connection.fetch(
                """
                SELECT node_id, attempt, sequence, request, response, started_at, finished_at
                FROM dagent_model_call WHERE run_id = $1
                ORDER BY node_id, attempt, sequence
                """,
                run_id,
            )

        return tuple(
            ModelCallRecord(
                run_id=run_id,
                node_id=row["node_id"],
                attempt=row["attempt"],
                sequence=row["sequence"],
                request=ModelRequest.model_validate_json(row["request"]),
                response=ModelResponse.model_validate_json(row["response"]),
                started_at=row["started_at"],
                finished_at=row["finished_at"],
            )
            for row in rows
        )

    # --- internals --------------------------------------------------------------------

    def _acquire(self) -> Any:
        """Borrow a connection from the pool for the duration of a block."""
        return self._pool.acquire()

    @staticmethod
    async def _write_node_state(
        connection: asyncpg.Connection[Any], record: NodeStateRecord
    ) -> None:
        """Upsert one node-state row on an already-borrowed connection."""
        await connection.execute(
            """
            INSERT INTO dagent_node_state
                (run_id, node_id, state, attempt, started_at, finished_at, error)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (run_id, node_id) DO UPDATE
            SET state       = EXCLUDED.state,
                attempt     = EXCLUDED.attempt,
                started_at  = EXCLUDED.started_at,
                finished_at = EXCLUDED.finished_at,
                error       = EXCLUDED.error
            """,
            record.run_id,
            record.node_id,
            record.state.value,
            record.attempt,
            record.started_at,
            record.finished_at,
            record.error,
        )
