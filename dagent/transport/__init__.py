"""The v2 work transport: the ``WorkQueue`` protocol and its implementations.

Everything about *how a ready node reaches the process that runs it* lives behind the
protocol, so the scheduler is transport-agnostic in exactly the way it is already
storage-agnostic (DR-3, DR-12). In-process in v1, Redis Streams in v2.

``RedisWorkQueue`` is deliberately not imported here: it needs the ``redis`` extra, and a
package that cannot be imported without an optional dependency is not optional.
"""

from dagent.transport.base import WorkItem, WorkQueue, WorkResult
from dagent.transport.memory import InMemoryWorkQueue

__all__ = ["REDIS_URL_ENV", "InMemoryWorkQueue", "WorkItem", "WorkQueue", "WorkResult"]

REDIS_URL_ENV = "DAGENT_REDIS_URL"
"""Where the Redis URL comes from.

Declared here rather than in ``redis.py`` for the same reason ``POSTGRES_DSN_ENV`` is
declared in ``store/__init__.py``: naming the transport must cost nothing, and importing
``redis.py`` pulls in an extra the base install does not have.
"""
