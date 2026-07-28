"""The two admission gates: concurrency permits and the run's budget.

Model APIs are rate-limited *and* cost money, so work is gated on two independent axes.
:class:`Limits` gates how many nodes may run at once — globally and per provider — and
:class:`Budget` gates how much a run may spend before it is cut off.

Both are deliberately ignorant of what they are gating: no agent, no model client, no
store. ``dagent.policy`` depends on ``dagent.errors`` and ``dagent.models`` and nothing
else, which is what keeps ``runtime`` free to depend on it without an import cycle.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager

from dagent.errors import PolicyError

__all__ = ["Budget", "Limits"]


class Limits:
    """Global and per-provider concurrency permits.

    A node acquires *both* a global permit and its provider's permit before it runs
    (DR-6). One global cap alone ignores that providers rate-limit independently; per
    provider alone lets a fan-out across three providers blow past the machine's total
    capacity. Acquiring both, always in the same order, is what makes the pair
    deadlock-free — a fixed order means two waiters can never each hold what the other
    needs.

    The gap between "the ready set is large" and "in-flight is capped" *is* the
    backpressure, and it is the reason the ready set can be recomputed eagerly without
    swamping a provider.
    """

    def __init__(
        self,
        *,
        max_concurrency: int | None = None,
        per_provider: Mapping[str, int] | None = None,
        default_per_provider: int | None = None,
    ) -> None:
        """Configure the caps. Every one of them defaults to unbounded.

        Unbounded by default so that wiring no limits reproduces the plain executor
        exactly, and a cap is something a run opts into rather than something it inherits.

        Args:
            max_concurrency: Nodes allowed to run at once across the whole run.
            per_provider: Explicit cap per provider name.
            default_per_provider: Cap applied to any provider not named above.

        Raises:
            PolicyError: If a cap is below one, which would stall the run forever.
        """
        _require_positive("max_concurrency", max_concurrency)
        _require_positive("default_per_provider", default_per_provider)
        for provider, cap in (per_provider or {}).items():
            _require_positive(f"per_provider[{provider!r}]", cap)

        self._per_provider = dict(per_provider or {})
        self._default_per_provider = default_per_provider
        self._global = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
        self._semaphores: dict[str, asyncio.Semaphore | None] = {}
        self._in_flight: dict[str, int] = {}
        self._peak: dict[str, int] = {}

    @asynccontextmanager
    async def slot(self, provider: str) -> AsyncGenerator[None, None]:
        """Hold a global permit and ``provider``'s permit for the body of the block.

        Both permits are released in ``finally``, including when the body is cancelled —
        a node that times out must not take its permits to the grave.
        """
        # Global first, then provider. The order is fixed rather than convenient: it is
        # what rules out the permit deadlock two-lock acquisition otherwise invites.
        async with _permit(self._global), _permit(self._provider_semaphore(provider)):
            self._in_flight[provider] = self._in_flight.get(provider, 0) + 1
            self._peak[provider] = max(self._peak.get(provider, 0), self._in_flight[provider])
            try:
                yield
            finally:
                self._in_flight[provider] -= 1

    def in_flight(self, provider: str) -> int:
        """How many nodes are inside a slot for this provider right now."""
        return self._in_flight.get(provider, 0)

    def peak(self, provider: str) -> int:
        """The most nodes this provider ever had in flight at once.

        Kept because it is the number the cap exists to hold down, so it is worth being
        able to assert on it — and it is the first metric Phase 7 will want to export.
        """
        return self._peak.get(provider, 0)

    def _provider_semaphore(self, provider: str) -> asyncio.Semaphore | None:
        """Return this provider's semaphore, creating it on first use.

        Lazily, because provider names are discovered at run time rather than declared:
        an unlisted provider gets ``default_per_provider``, and ``None`` means unbounded.
        """
        if provider not in self._semaphores:
            cap = self._per_provider.get(provider, self._default_per_provider)
            self._semaphores[provider] = asyncio.Semaphore(cap) if cap is not None else None
        return self._semaphores[provider]


class Budget:
    """A run's token and cost ceiling, and the admission control that enforces it.

    Holds numbers and nothing else — it never sees a request, a response, or a provider.
    The wrapper that charges it lives in ``dagent.runtime.metering``, which is what keeps
    this module free of any dependency on the model seam.
    """

    def __init__(self, *, max_tokens: int | None = None, max_cost_usd: float | None = None) -> None:
        """Set the ceilings. Both default to unbounded.

        Args:
            max_tokens: Total tokens, input plus output, the run may spend.
            max_cost_usd: Total cost the run may spend. See
                :data:`~dagent.runtime.metering.free` for why no price table ships here.

        Raises:
            PolicyError: If a ceiling is negative.
        """
        if max_tokens is not None and max_tokens < 0:
            raise PolicyError(f"max_tokens must not be negative, got {max_tokens}")
        if max_cost_usd is not None and max_cost_usd < 0:
            raise PolicyError(f"max_cost_usd must not be negative, got {max_cost_usd}")

        self._max_tokens = max_tokens
        self._max_cost_usd = max_cost_usd
        self._tokens_used = 0
        self._cost_used = 0.0
        self._refusals = 0

    @property
    def tokens_used(self) -> int:
        """Tokens charged to this run so far."""
        return self._tokens_used

    @property
    def cost_used(self) -> float:
        """Cost charged to this run so far, in USD."""
        return self._cost_used

    @property
    def refusals(self) -> int:
        """How many calls this budget has refused to admit."""
        return self._refusals

    @property
    def refused(self) -> bool:
        """Whether this budget has actually stopped work.

        Distinct from :attr:`exceeded` on purpose. A ceiling crossed by the very last call
        of an otherwise complete run did not stop anything, and calling that run
        ``BUDGET_EXCEEDED`` would misreport a success. A run is ``BUDGET_EXCEEDED`` when
        the budget *refused* something.
        """
        return self._refusals > 0

    @property
    def exceeded(self) -> bool:
        """Whether usage has reached a ceiling."""
        if self._max_tokens is not None and self._tokens_used >= self._max_tokens:
            return True
        return self._max_cost_usd is not None and self._cost_used >= self._max_cost_usd

    def admit(self) -> None:
        """Allow one model call, or refuse it.

        Raises:
            PolicyError: If a ceiling has been reached. The refusal is counted, which is
                what later turns the run's outcome into ``BUDGET_EXCEEDED``.
        """
        if self.exceeded:
            self._refusals += 1
            raise PolicyError(f"budget exceeded ({self.describe()}); no further model calls")

    def charge(self, *, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Record what a completed call consumed.

        Charged after the fact rather than reserved up front: nobody knows how many tokens
        a completion will produce until it has produced them, so the ceiling is enforced by
        refusing the *next* call rather than by pretending to predict this one.
        """
        self._tokens_used += tokens
        self._cost_used += cost_usd

    def describe(self) -> str:
        """One-line summary of usage against the ceilings, for error messages and reports."""
        tokens = f"{self._tokens_used}/{self._max_tokens if self._max_tokens is not None else '∞'}"
        parts = [f"tokens {tokens}"]
        if self._max_cost_usd is not None:
            parts.append(f"cost ${self._cost_used:.4f}/${self._max_cost_usd:.4f}")
        return ", ".join(parts)


@asynccontextmanager
async def _permit(semaphore: asyncio.Semaphore | None) -> AsyncGenerator[None, None]:
    """Hold one permit, or nothing at all when that axis is unbounded."""
    if semaphore is None:
        yield
        return

    await semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def _require_positive(name: str, cap: int | None) -> None:
    """Reject a cap that would stall every node forever."""
    if cap is not None and cap < 1:
        raise PolicyError(f"{name} must be at least 1, got {cap}")
