"""The `queue` fixture: every transport test runs against every implementation.

Same argument as `tests/conftest.py` makes for the store. "The transport is swappable" is
only worth claiming if two of them are actually held to one contract, and the only honest
way to hold them is to run the same tests against both rather than writing two suites that
drift apart.

Redis needs a live server, which a unit test suite must not require. Set
``DAGENT_TEST_REDIS_URL`` and the Redis parametrisation runs; leave it unset and it is
skipped with a reason, so nobody mistakes "not run" for "passed".

Each Redis test gets its own key prefix and deletes it afterwards. Streams are persistent
and consumer groups remember every consumer that ever joined, so a shared prefix would make
one test's pending entries into the next test's mystery failure.
"""

import os
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from dagent.transport.base import WorkQueue
from dagent.transport.memory import InMemoryWorkQueue

REDIS_URL_ENV = "DAGENT_TEST_REDIS_URL"


@pytest_asyncio.fixture(params=["memory", "redis"])
async def queue(request: pytest.FixtureRequest) -> AsyncIterator[WorkQueue]:
    """Yield a clean work queue of each implementation in turn."""
    if request.param == "memory":
        yield InMemoryWorkQueue()
        return

    url = os.environ.get(REDIS_URL_ENV)
    if not url:
        pytest.skip(f"{REDIS_URL_ENV} is not set; skipping the Redis transport")

    from dagent.transport.redis import RedisWorkQueue

    prefix = f"dagent-test-{uuid.uuid4().hex[:12]}"
    redis_queue = await RedisWorkQueue.connect(url, prefix=prefix)
    try:
        yield redis_queue
    finally:
        keys = await redis_queue._redis.keys(f"{prefix}:*")
        if keys:
            await redis_queue._redis.delete(*keys)
        await redis_queue.close()
