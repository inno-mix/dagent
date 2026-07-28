"""The exception hierarchy for dagent.

Every error this package raises derives from :class:`DagentError`, so a caller can catch
one type and know the failure came from the engine rather than from a dependency. Bare
``Exception`` is never raised and never swallowed (AGENTS.md §5).
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "AgentError",
    "DagentError",
    "PolicyError",
    "StoreError",
    "ValidationError",
]


class DagentError(Exception):
    """Base class for every error raised by dagent."""


class ValidationError(DagentError):
    """A workflow definition was rejected before any execution began.

    Why this is its own type: validation is total and happens at submit time, so a caller
    can treat it as "this workflow will never run" rather than as a transient failure.
    """

    def __init__(self, message: str, *, cycle: Sequence[str] | None = None) -> None:
        """Record the rejection, optionally with the dependency cycle that caused it.

        Args:
            message: Human-readable explanation, already naming the offending nodes.
            cycle: The cycle path, when the rejection was a cycle. Each element depends
                on the next, and the first and last elements are the same node.
        """
        super().__init__(message)
        self.cycle: tuple[str, ...] | None = tuple(cycle) if cycle is not None else None


class PolicyError(DagentError):
    """A retry, timeout, concurrency, or budget rule stopped execution."""


class AgentError(DagentError):
    """An agent raised while executing a node.

    Carries a ``retryable`` hint because FR-5 retries only errors *classified* retryable,
    and the code best placed to classify a failure is the code that raised it: a provider
    client knows a 429 is weather and a 400 is a bug in the request. The default is
    ``True`` — most agent failures are transient provider trouble — so a permanent failure
    says so explicitly, and a caller who disagrees with the whole scheme passes its own
    classifier to the policy layer.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        """Record the failure and whether another attempt could plausibly succeed.

        Args:
            message: Human-readable explanation, already naming the node or provider.
            retryable: ``False`` when the same call would fail the same way again.
        """
        super().__init__(message)
        self.retryable = retryable


class StoreError(DagentError):
    """A ``StateStore`` operation failed."""
