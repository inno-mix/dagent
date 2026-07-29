"""The `store` fixture: every store test runs against every implementation.

Phase 5's acceptance says crash-resume must pass "against both memory and Postgres
stores", and the only honest way to claim that is to run the same tests against both
rather than to write two suites that drift apart.

Postgres needs a live database, which a unit test suite must not require. Set
``DAGENT_TEST_POSTGRES_DSN`` and the Postgres parametrisation runs; leave it unset and it
is skipped with a reason, so nobody mistakes "not run" for "passed".
"""

import os

import pytest
import pytest_asyncio

from dagent.store.memory import InMemoryStateStore

POSTGRES_DSN_ENV = "DAGENT_TEST_POSTGRES_DSN"

TABLES = (
    "dagent_model_call",
    "dagent_output",
    "dagent_node_state",
    "dagent_workflow",
    "dagent_run",
)


@pytest_asyncio.fixture(params=["memory", "postgres"])
async def store(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    """Yield a clean store of each implementation in turn."""
    if request.param == "memory":
        yield InMemoryStateStore()
        return

    dsn = os.environ.get(POSTGRES_DSN_ENV)
    if not dsn:
        pytest.skip(f"{POSTGRES_DSN_ENV} is not set; skipping the Postgres store")

    from dagent.store.postgres import PostgresStateStore

    postgres = await PostgresStateStore.connect(dsn)
    # Truncate rather than drop: the schema is what `connect` just asserted, and a test
    # that starts from an empty table proves isolation without re-running DDL each time.
    async with postgres._pool.acquire() as connection:
        await connection.execute(f"TRUNCATE {', '.join(TABLES)} CASCADE")
    try:
        yield postgres
    finally:
        await postgres.close()
