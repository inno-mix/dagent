"""The ``StateStore`` protocol and its implementations.

Every durability semantic lives behind the protocol so the executor stays storage
agnostic: in-memory in v1, Postgres in v2 (DR-3).
"""

from dagent.store.base import StateStore
from dagent.store.memory import InMemoryStateStore

__all__ = ["InMemoryStateStore", "StateStore"]
