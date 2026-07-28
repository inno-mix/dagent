"""Retry policy: attempts, backoff, and which failures earn another try.

Pure computation over the node's :class:`~dagent.models.workflow.Policy` — nothing here
sleeps, dispatches, or touches a store. The executor asks this module *how long* and
*whether*, then does the waiting through the injected ``Clock``, which is what keeps the
retry path replayable (AGENTS.md rule 4).

The one source of non-determinism, jitter, is a seam rather than a call: :data:`Jitter` is
an ordinary function, :func:`full_jitter` is the production one, and :func:`no_jitter` is
what a test or a replay injects to get the same delays twice.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

from dagent.errors import AgentError
from dagent.models.workflow import Policy

__all__ = [
    "Backoff",
    "Jitter",
    "Retryable",
    "default_retryable",
    "full_jitter",
    "no_jitter",
]

Jitter: TypeAlias = Callable[[float], float]
"""Given the backoff ceiling for an attempt, return the delay to actually wait."""

Retryable: TypeAlias = Callable[[BaseException], bool]
"""Given the exception a node raised, decide whether another attempt is worth making."""

_MAX_DOUBLINGS = 32
"""Where exponential growth stops mattering.

``backoff_initial_s * 2**attempt`` overflows a float long before a real workflow reaches
that many attempts, and the result is clamped to ``backoff_max_s`` anyway — so the shift
is capped rather than left to raise ``OverflowError`` on an absurd input.
"""


def full_jitter(ceiling: float) -> float:
    """Return a uniform sample in ``[0, ceiling]``.

    "Full jitter" rather than a fixed exponential delay: when several nodes fail against
    the same rate-limited provider at the same instant, identical backoff makes them all
    retry at the same instant too, and the thundering herd re-creates the overload that
    caused the failure. Spreading the retries out is the whole point.
    """
    return random.uniform(0.0, ceiling)


def no_jitter(ceiling: float) -> float:
    """Return the ceiling unchanged — the deterministic seam for tests and replay."""
    return ceiling


def default_retryable(exc: BaseException) -> bool:
    """Classify a node failure as worth another attempt.

    Retryable: an :class:`~dagent.errors.AgentError` that did not mark itself permanent,
    and a ``TimeoutError`` — the two shapes a provider hiccup arrives in.

    Everything else is not, on purpose. A ``ValidationError`` will be exactly as true next
    time; a ``PolicyError`` means a limit already refused this work and retrying is how
    you exceed it twice; a ``StoreError`` means the thing recording the attempt is broken;
    and a ``TypeError`` is a bug, which retrying turns into three identical stack traces
    and three times the latency.
    """
    if isinstance(exc, AgentError):
        return exc.retryable
    return isinstance(exc, TimeoutError)


@dataclass(frozen=True, slots=True)
class Backoff:
    """Exponential backoff with jitter, computed from a node's policy."""

    jitter: Jitter = field(default=full_jitter)

    def delay(self, policy: Policy, attempt: int) -> float:
        """Return how long to wait before the attempt after ``attempt``.

        Args:
            policy: The node's policy, supplying the initial delay and the ceiling.
            attempt: The zero-based attempt that just failed, so the first retry waits
                roughly ``backoff_initial_s`` and each subsequent one doubles.

        Returns:
            Seconds to wait, never above ``policy.backoff_max_s``.
        """
        ceiling = min(
            policy.backoff_initial_s * float(2 ** min(attempt, _MAX_DOUBLINGS)),
            policy.backoff_max_s,
        )
        return self.jitter(ceiling)
