"""Redis Streams as the work transport (Phase 8, AGENTS.md §4).

Streams rather than lists or pub/sub, for one reason each. A list (``BRPOP``) hands a
message to a consumer and forgets it, so a worker that dies mid-node takes the node with
it — there is nothing left to redeliver. Pub/sub is worse: it drops messages nobody is
listening for. A stream plus a consumer group keeps a delivered-but-unacknowledged message
in a *pending entries list*, which is exactly the state "somebody is working on this, and
may not finish". ``XAUTOCLAIM`` is then the whole of failure recovery: take the entries
that have gone stale and run them again.

Two channels, matching :mod:`dagent.transport.base`:

* ``{prefix}:work`` — one stream, group ``{prefix}:workers``. Every worker joins the same
  group, so Redis hands each entry to exactly one of them.
* ``{prefix}:results:{run_id}`` — one stream per run, group ``{prefix}:coordinator``, read
  by the single coordinator that owns that run.

Result streams are given a TTL rather than being deleted, so a run abandoned halfway does
not leak a key forever and a coordinator that comes back to resume within the window still
finds everything it never acknowledged.

This is the only module in dagent that imports ``redis``; it ships under the ``redis``
extra, and ``tests/test_isolation.py`` fails the build if it leaks anywhere else.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from dagent.errors import TransportError
from dagent.models.state import NodeState
from dagent.models.workflow import Node
from dagent.transport.base import WorkItem, WorkResult

__all__ = ["RedisWorkQueue"]

DEFAULT_PREFIX = "dagent"
RESULT_TTL_S = 24 * 60 * 60
"""How long a run's results channel outlives its last message.

Long enough that a crashed coordinator can be resumed the next morning and still find the
completions it never acknowledged; short enough that a run nobody ever comes back to
disappears on its own.
"""


class RedisWorkQueue:
    """A ``WorkQueue`` backed by Redis Streams and consumer groups.

    Construct with :meth:`connect`, which also creates the work group — every other method
    then assumes it exists, rather than each one re-checking on a hot path.
    """

    def __init__(self, client: Redis, *, prefix: str = DEFAULT_PREFIX) -> None:
        """Wrap an already-connected client.

        Prefer :meth:`connect` unless you are sharing a client, in which case the caller
        owns closing it — :meth:`close` closes whatever it was given.
        """
        self._redis = client
        self._prefix = prefix
        self._work = f"{prefix}:work"
        self._workers = f"{prefix}:workers"
        self._coordinator = f"{prefix}:coordinator"
        self._followed: set[str] = set()

    @classmethod
    async def connect(cls, url: str, *, prefix: str = DEFAULT_PREFIX) -> RedisWorkQueue:
        """Open a connection and make sure the work group exists.

        Raises:
            TransportError: If Redis is unreachable.
        """
        client: Redis = Redis.from_url(url, decode_responses=True)
        queue = cls(client, prefix=prefix)
        try:
            await queue._ensure_group(queue._work, queue._workers)
        except TransportError:
            await client.aclose()
            raise
        return queue

    # --- the coordinator's half ---------------------------------------------------------

    async def follow(self, run_id: str) -> None:
        """Create this run's results group, from the beginning of its stream.

        From ``0`` rather than from ``$``: a group created at "only what arrives next"
        would silently skip every result a previous coordinator had already been sent and
        never acknowledged, which is the exact set ``resume`` exists to pick up.
        """
        if run_id in self._followed:
            return
        await self._ensure_group(self._results(run_id), self._coordinator)
        self._followed.add(run_id)

    async def submit(self, item: WorkItem) -> None:
        """Append a node to the shared work stream."""
        await self._command(
            self._redis.xadd(
                self._work,
                {"run_id": item.run_id, "node_id": item.node_id, "attempt": str(item.attempt)},
            )
        )

    async def collect(self, run_id: str, *, timeout_s: float = 1.0) -> Sequence[WorkResult]:
        """Read this run's results, taking anything still unacknowledged first."""
        # Cheap after the first call, and it means reading a channel nobody followed yet
        # returns nothing rather than failing — the same thing the in-memory queue does.
        await self.follow(run_id)
        stream = self._results(run_id)
        reclaimed = await self._command(
            self._redis.xautoclaim(
                stream, self._coordinator, self._coordinator, min_idle_time=0, count=64
            )
        )
        entries = _autoclaimed(reclaimed)
        if entries:
            return tuple(_result(run_id, message_id, fields) for message_id, fields in entries)

        response = await self._command(
            self._redis.xreadgroup(
                self._coordinator,
                self._coordinator,
                {stream: ">"},
                count=64,
                block=max(int(timeout_s * 1000), 1),
            )
        )
        return tuple(
            _result(run_id, message_id, fields)
            for _, messages in response or ()
            for message_id, fields in messages
        )

    async def settle(self, results: Sequence[WorkResult]) -> None:
        """Acknowledge results, and refresh the stream's TTL while we are here."""
        by_run: dict[str, list[str]] = {}
        for result in results:
            by_run.setdefault(result.run_id, []).append(result.receipt)
        for run_id, receipts in by_run.items():
            stream = self._results(run_id)
            await self._command(self._redis.xack(stream, self._coordinator, *receipts))
            await self._command(self._redis.expire(stream, RESULT_TTL_S))

    # --- a worker's half ----------------------------------------------------------------

    async def claim(self, *, consumer: str, timeout_s: float = 1.0) -> WorkItem | None:
        """Block on the work stream until an entry arrives or the timeout passes."""
        response = await self._command(
            self._redis.xreadgroup(
                self._workers,
                consumer,
                {self._work: ">"},
                count=1,
                block=max(int(timeout_s * 1000), 1),
            )
        )
        for _, messages in response or ():
            for message_id, fields in messages:
                return _item(message_id, fields)
        return None

    async def reclaim(self, *, consumer: str, min_idle_s: float) -> Sequence[WorkItem]:
        """Take over entries this group delivered and nobody acknowledged.

        ``XAUTOCLAIM`` keeps the message id, so the entry is transferred rather than
        reissued: whichever worker finishes first acknowledges the same id, and the other
        one's acknowledgement is a no-op instead of retiring a second copy.
        """
        response = await self._command(
            self._redis.xautoclaim(
                self._work,
                self._workers,
                consumer,
                min_idle_time=max(int(min_idle_s * 1000), 0),
                count=16,
            )
        )
        return tuple(_item(message_id, fields) for message_id, fields in _autoclaimed(response))

    async def complete(self, item: WorkItem, result: WorkResult) -> None:
        """Publish the result, then acknowledge the work entry — in that order."""
        stream = self._results(result.run_id)
        await self._command(
            self._redis.xadd(
                stream,
                {
                    "run_id": result.run_id,
                    "node_id": result.node_id,
                    "attempt": str(result.attempt),
                    "state": result.state.value,
                    "expansion": json.dumps(
                        [node.model_dump(mode="json") for node in result.expansion]
                    ),
                },
            )
        )
        await self._command(self._redis.expire(stream, RESULT_TTL_S))
        await self._command(self._redis.xack(self._work, self._workers, item.receipt))

    async def close(self) -> None:
        """Close the client this queue was given."""
        await self._redis.aclose()

    # --- plumbing -----------------------------------------------------------------------

    def _results(self, run_id: str) -> str:
        """The stream one run's completions are published to."""
        return f"{self._prefix}:results:{run_id}"

    async def _ensure_group(self, stream: str, group: str) -> None:
        """Create a consumer group, tolerating one that already exists.

        Idempotent by design rather than by check-then-create: two coordinators starting at
        once would both see "no group" and both try to create it, and the loser of that
        race is not an error — it is the group being there, which is what was wanted.
        """
        try:
            await self._redis.xgroup_create(stream, group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise TransportError(f"could not create consumer group {group!r}: {exc}") from exc
        except RedisError as exc:
            raise TransportError(f"redis is unreachable: {exc}") from exc

    async def _command(self, awaitable: Any) -> Any:
        """Await one Redis call, translating its failures into the dagent hierarchy."""
        try:
            return await awaitable
        except RedisError as exc:
            raise TransportError(f"redis command failed: {exc}") from exc


def _item(message_id: str, fields: Mapping[str, str]) -> WorkItem:
    """Rebuild a work item from a stream entry."""
    return WorkItem(
        run_id=fields["run_id"],
        node_id=fields["node_id"],
        attempt=int(fields["attempt"]),
        receipt=message_id,
    )


def _result(run_id: str, message_id: str, fields: Mapping[str, str]) -> WorkResult:
    """Rebuild a result from a stream entry.

    The expansion is carried as JSON of the node models rather than as a reference, because
    the coordinator has to validate nodes that exist nowhere else yet — they are precisely
    the ones not in the stored graph.
    """
    return WorkResult(
        run_id=run_id,
        node_id=fields["node_id"],
        attempt=int(fields["attempt"]),
        state=NodeState(fields["state"]),
        expansion=tuple(Node.model_validate(node) for node in json.loads(fields["expansion"])),
        receipt=message_id,
    )


def _autoclaimed(response: Any) -> Sequence[tuple[str, Mapping[str, str]]]:
    """Pull the entries out of an ``XAUTOCLAIM`` reply.

    Redis 7 answers with ``(cursor, entries, deleted)`` and Redis 6.2 with
    ``(cursor, entries)``. Indexing position 1 rather than unpacking is what keeps this
    working on both without a version check.
    """
    if not response or len(response) < 2:
        return ()
    entries: Sequence[tuple[str, Mapping[str, str]]] = response[1]
    return entries
