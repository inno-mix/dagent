"""The ``StateStore`` protocol and its implementations.

Every durability semantic lives behind the protocol so the executor stays storage
agnostic: in-memory in v1, Postgres in v2 (DR-3).
"""

from dagent.store.base import StateStore
from dagent.store.memory import InMemoryStateStore

__all__ = ["POSTGRES_DSN_ENV", "InMemoryStateStore", "StateStore"]

POSTGRES_DSN_ENV = "DAGENT_POSTGRES_DSN"
"""Where the Postgres DSN comes from.

Declared here rather than in ``postgres.py`` so that naming the store costs nothing:
importing ``postgres.py`` pulls in ``asyncpg``, which is an optional extra, and the CLI
has to be able to *offer* the Postgres store while running perfectly well without it
installed.
"""
