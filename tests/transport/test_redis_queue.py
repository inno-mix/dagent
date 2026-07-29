"""Redis-specific behaviour: connecting, and failing loudly when Redis is not there.

The queue's *contract* is tested in `test_queue_conformance.py` against every
implementation. What is left here is the part only this implementation has.
"""

import os

import pytest

from dagent.errors import TransportError
from dagent.transport import REDIS_URL_ENV

URL = os.environ.get("DAGENT_TEST_REDIS_URL")
needs_redis = pytest.mark.skipif(
    not URL, reason="DAGENT_TEST_REDIS_URL is not set; skipping the Redis transport"
)


def test_the_url_variable_has_exactly_one_definition() -> None:
    # `cli.py` names the variable without importing `transport.redis`, because importing it
    # needs the extra. Two spellings that drift apart is the failure that guards.
    assert REDIS_URL_ENV == "DAGENT_REDIS_URL"


@pytest.mark.asyncio
async def test_connecting_to_a_broker_that_is_not_there_raises_a_transport_error() -> None:
    # A refused connection is a TransportError like any other, so a caller catching
    # DagentError does not have to know redis-py's exception hierarchy exists.
    from dagent.transport.redis import RedisWorkQueue

    with pytest.raises(TransportError, match="redis"):
        await RedisWorkQueue.connect("redis://127.0.0.1:1/0")


@needs_redis
@pytest.mark.asyncio
async def test_a_prefix_keeps_two_deployments_out_of_each_others_streams() -> None:
    # Two dagent installations sharing one Redis must not consume each other's work.
    assert URL is not None
    from dagent.transport.base import WorkItem
    from dagent.transport.redis import RedisWorkQueue

    mine = await RedisWorkQueue.connect(URL, prefix="dagent-test-mine")
    theirs = await RedisWorkQueue.connect(URL, prefix="dagent-test-theirs")
    try:
        await mine.submit(WorkItem("r1", "a", 0))

        assert await theirs.claim(consumer="w1", timeout_s=0.05) is None
        assert await mine.claim(consumer="w1", timeout_s=0.05) is not None
    finally:
        for queue, prefix in ((mine, "dagent-test-mine"), (theirs, "dagent-test-theirs")):
            keys = await queue._redis.keys(f"{prefix}:*")
            if keys:
                await queue._redis.delete(*keys)
            await queue.close()
