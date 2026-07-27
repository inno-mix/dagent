import dataclasses

import pytest

from dagent.runtime.agent import AgentContext
from dagent.runtime.clock import ManualClock


def a_context(**overrides: object) -> AgentContext:
    fields: dict[str, object] = {
        "run_id": "r1",
        "node_id": "a",
        "attempt": 0,
        "inputs": {},
        "clock": ManualClock(),
    }
    fields.update(overrides)
    return AgentContext(**fields)  # type: ignore[arg-type]


def test_context_carries_the_run_and_node_identity() -> None:
    ctx = a_context()

    assert (ctx.run_id, ctx.node_id) == ("r1", "a")


def test_context_carries_resolved_inputs() -> None:
    ctx = a_context(inputs={"upstream": {"summary": "hi"}})

    assert ctx.inputs["upstream"] == {"summary": "hi"}


def test_context_carries_the_attempt_for_the_phase_5_idempotency_key() -> None:
    ctx = a_context(attempt=2)

    assert (ctx.run_id, ctx.node_id, ctx.attempt) == ("r1", "a", 2)


def test_context_is_frozen() -> None:
    # An agent must not be able to rewrite its own identity mid-run.
    ctx = a_context()

    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.node_id = "b"  # type: ignore[misc]


def test_the_clock_reaches_the_agent_through_the_context() -> None:
    clock = ManualClock()
    ctx = a_context(clock=clock)

    assert ctx.clock.now() == clock.now()
