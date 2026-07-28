"""Recording every model call into the run record — the foundation for replay.

FR-8 wants a run reproducible offline. That needs two halves: capture, and playback.
This is capture. It wraps any :class:`~dagent.runtime.model.ModelClient` and writes each
request and response into the store as a :class:`ModelCallRecord`, keyed so playback can
find them again — ``(run_id, node_id, attempt, sequence)``.

The wrapper is per-node and cheap; the client it wraps is shared for the whole run, which
is how "one shared ``httpx.AsyncClient``, never per-call" (AGENTS.md §4) survives being
wrapped.
"""

from __future__ import annotations

from dagent.models.model_call import ModelCallRecord, ModelRequest, ModelResponse
from dagent.runtime.clock import Clock
from dagent.runtime.model import ModelClient
from dagent.store.base import StateStore

__all__ = ["RecordingModelClient"]


class RecordingModelClient:
    """A ``ModelClient`` that persists every call it makes on behalf of one node."""

    def __init__(
        self,
        inner: ModelClient,
        *,
        store: StateStore,
        run_id: str,
        node_id: str,
        attempt: int = 0,
        clock: Clock | None = None,
    ) -> None:
        """Wrap ``inner``, tagging everything it records with this node's identity."""
        self._inner = inner
        self._store = store
        self._run_id = run_id
        self._node_id = node_id
        self._attempt = attempt
        self._clock = clock
        self._sequence = 0

    @property
    def provider(self) -> str:
        """The wrapped client's provider — recording does not change who answers."""
        return self._inner.provider

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Make the call, record it, and return the response unchanged.

        Recording happens after the response arrives, so a failed call records nothing:
        there is no response to replay, and the retry that follows (Phase 4) will make
        its own call under a new attempt number.
        """
        started = self._clock.now() if self._clock is not None else None
        response = await self._inner.complete(request)
        finished = self._clock.now() if self._clock is not None else None

        await self._store.append_model_call(
            ModelCallRecord(
                run_id=self._run_id,
                node_id=self._node_id,
                attempt=self._attempt,
                sequence=self._sequence,
                request=request,
                response=response,
                started_at=started,
                finished_at=finished,
            )
        )
        self._sequence += 1
        return response
