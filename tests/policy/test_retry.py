"""Backoff arithmetic and retryable classification — the two decisions retry rests on."""

import pytest

from dagent.errors import (
    AgentError,
    PolicyError,
    StoreError,
    ValidationError,
)
from dagent.models.workflow import Policy
from dagent.policy.retry import (
    Backoff,
    default_retryable,
    full_jitter,
    no_jitter,
)

DOUBLING = Policy(backoff_initial_s=1.0, backoff_max_s=1000.0)


def test_backoff_doubles_with_each_attempt() -> None:
    backoff = Backoff(jitter=no_jitter)

    delays = [backoff.delay(DOUBLING, attempt) for attempt in range(5)]

    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_backoff_is_clamped_to_the_policy_ceiling() -> None:
    policy = Policy(backoff_initial_s=1.0, backoff_max_s=5.0)
    backoff = Backoff(jitter=no_jitter)

    assert [backoff.delay(policy, attempt) for attempt in range(5)] == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_an_absurd_attempt_number_clamps_rather_than_overflowing() -> None:
    # 1.0 * 2**5000 raises OverflowError; the shift is capped so a runaway attempt
    # counter degrades to "wait the maximum" instead of crashing the retry path.
    assert Backoff(jitter=no_jitter).delay(DOUBLING, 5000) == 1000.0


def test_full_jitter_stays_inside_the_ceiling() -> None:
    samples = [full_jitter(4.0) for _ in range(200)]

    assert all(0.0 <= sample <= 4.0 for sample in samples)


def test_full_jitter_actually_spreads_the_delays_out() -> None:
    # The point of jitter is that two nodes failing at the same instant do not retry at
    # the same instant. A jitter that returned a constant would pass every other test.
    samples = {full_jitter(10.0) for _ in range(50)}

    assert len(samples) > 1


def test_the_jitter_seam_makes_backoff_reproducible() -> None:
    backoff = Backoff(jitter=lambda ceiling: ceiling / 2)

    assert backoff.delay(DOUBLING, 2) == 2.0


def test_full_jitter_is_the_default() -> None:
    # Production must not silently get the deterministic one.
    assert Backoff().jitter is full_jitter


# --- classification ------------------------------------------------------------------


def test_an_agent_error_is_retryable_by_default() -> None:
    # Most agent failures are a provider having a moment.
    assert default_retryable(AgentError("provider hiccup")) is True


def test_an_agent_error_can_declare_itself_permanent() -> None:
    assert default_retryable(AgentError("no such model", retryable=False)) is False


def test_a_timeout_is_retryable() -> None:
    assert default_retryable(TimeoutError()) is True


@pytest.mark.parametrize(
    "error",
    [
        ValidationError("this graph will never be valid"),
        PolicyError("budget exceeded"),
        StoreError("the store is broken"),
        TypeError("this is a bug"),
        ValueError("so is this"),
        RuntimeError("and this"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_permanent_failures_are_not_retried(error: BaseException) -> None:
    # Retrying a bug produces three identical stack traces and three times the latency.
    assert default_retryable(error) is False


def test_no_jitter_returns_the_ceiling_unchanged() -> None:
    assert no_jitter(3.5) == 3.5
