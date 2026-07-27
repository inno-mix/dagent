"""The ``StateStore`` protocol and its implementations.

Every durability semantic lives behind the protocol so the executor stays storage
agnostic: in-memory in v1, Postgres in v2. Arrives in Phase 2.
"""
