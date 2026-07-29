"""Postgres-specific behaviour: connecting, the context manager, and failing loudly.

The store's *contract* is tested in `test_conformance.py` against every implementation.
What is left here is the part only this implementation has — a connection to open, a pool
to close, and a database that might not be there.

Skipped wholesale without `DAGENT_TEST_POSTGRES_DSN`, so nobody mistakes "not run" for
"passed".
"""

import os

import pytest

from dagent.errors import StoreError
from dagent.store import POSTGRES_DSN_ENV
from dagent.store.base import StateStore
from dagent.store.postgres import DSN_ENV, PostgresStateStore

DSN = os.environ.get("DAGENT_TEST_POSTGRES_DSN")
needs_postgres = pytest.mark.skipif(
    not DSN, reason="DAGENT_TEST_POSTGRES_DSN is not set; skipping the Postgres store"
)


def test_the_dsn_variable_has_exactly_one_definition() -> None:
    # `cli.py` names the variable without importing this module, because importing this
    # module needs asyncpg. Two spellings that drift apart is the failure that guards.
    assert DSN_ENV == POSTGRES_DSN_ENV == "DAGENT_POSTGRES_DSN"


@pytest.mark.asyncio
async def test_connecting_to_a_database_that_is_not_there_raises_a_store_error() -> None:
    # A refused connection is a StoreError like any other, so a caller catching
    # DagentError does not have to know asyncpg's exception hierarchy exists.
    with pytest.raises(StoreError, match="cannot connect to Postgres"):
        await PostgresStateStore.connect("postgresql://nobody:nobody@127.0.0.1:1/none", timeout=2)


@needs_postgres
@pytest.mark.asyncio
async def test_connect_creates_the_schema_and_satisfies_the_protocol() -> None:
    assert DSN is not None
    store = await PostgresStateStore.connect(DSN)
    try:
        assert isinstance(store, StateStore)
        # Idempotent: connecting twice against a live schema must not fail.
        await store.create_schema()
    finally:
        await store.close()


@needs_postgres
@pytest.mark.asyncio
async def test_the_context_manager_closes_the_pool() -> None:
    assert DSN is not None
    store = await PostgresStateStore.connect(DSN)

    async with store:
        pass

    with pytest.raises(Exception, match="closed"):
        await store.load_run("anything")
