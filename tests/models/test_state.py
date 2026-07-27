from datetime import UTC, datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from dagent.models.state import (
    TERMINAL_NODE_STATES,
    TERMINAL_RUN_STATES,
    NodeState,
    NodeStateRecord,
    RunState,
    RunStateRecord,
)


def test_node_state_members_match_architecture_md() -> None:
    assert set(NodeState.__members__) == {
        "PENDING",
        "READY",
        "RUNNING",
        "SUCCESS",
        "FAILED",
        "SKIPPED",
    }


def test_run_state_members_match_architecture_md() -> None:
    assert set(RunState.__members__) == {
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "BUDGET_EXCEEDED",
        "CANCELLED",
    }


def test_node_state_and_run_state_are_distinct_types() -> None:
    # AGENTS.md §5 names conflating these as the most common bug in this codebase.
    assert NodeState.PENDING is not RunState.PENDING
    assert not isinstance(NodeState.PENDING, RunState)
    assert not isinstance(RunState.PENDING, NodeState)


def test_the_two_enums_overlap_only_where_the_meaning_genuinely_does() -> None:
    shared = set(NodeState.__members__) & set(RunState.__members__)

    assert shared == {"PENDING", "RUNNING", "FAILED"}
    # A node SUCCEEDs, a run SUCCEEDEDs; the asymmetry is deliberate, not a typo.
    assert "SUCCESS" not in RunState.__members__
    assert "SUCCEEDED" not in NodeState.__members__


def test_terminal_node_states_are_the_ones_no_transition_leaves() -> None:
    expected = {NodeState.SUCCESS, NodeState.FAILED, NodeState.SKIPPED}

    assert set(TERMINAL_NODE_STATES) == expected


def test_terminal_run_states_are_the_ones_no_transition_leaves() -> None:
    expected = {
        RunState.SUCCEEDED,
        RunState.FAILED,
        RunState.BUDGET_EXCEEDED,
        RunState.CANCELLED,
    }

    assert set(TERMINAL_RUN_STATES) == expected


def test_node_state_record_defaults_are_the_start_of_a_lifecycle() -> None:
    record = NodeStateRecord(run_id="run-1", node_id="a")

    assert record.state is NodeState.PENDING
    assert record.attempt == 0
    assert record.error is None


def test_records_have_no_wall_clock_defaults() -> None:
    # AGENTS.md rule 4: non-determinism goes through an injected seam, never a default.
    # Two records built from the same arguments must be indistinguishable.
    first = NodeStateRecord(run_id="run-1", node_id="a")
    second = NodeStateRecord(run_id="run-1", node_id="a")

    assert first.started_at is None
    assert first.finished_at is None
    assert first == second


def test_run_state_record_defaults() -> None:
    run = RunStateRecord(run_id="run-1", workflow_name="research")

    assert run.state is RunState.PENDING
    assert run.created_at is None
    assert run.updated_at is None
    assert dict(run.nodes) == {}


def test_records_accept_timestamps_supplied_by_the_caller() -> None:
    at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    record = NodeStateRecord(run_id="run-1", node_id="a", started_at=at)

    assert record.started_at == at


def test_attempt_supports_the_phase_5_idempotency_key() -> None:
    # DR-4: (run_id, node_id, attempt) must be derivable without a schema change.
    record = NodeStateRecord(run_id="run-1", node_id="a", attempt=2)

    assert (record.run_id, record.node_id, record.attempt) == ("run-1", "a", 2)


def test_attempt_cannot_be_negative() -> None:
    with pytest.raises(PydanticValidationError):
        NodeStateRecord(run_id="run-1", node_id="a", attempt=-1)


def test_records_are_frozen() -> None:
    record = NodeStateRecord(run_id="run-1", node_id="a")

    with pytest.raises(PydanticValidationError):
        record.state = NodeState.RUNNING  # type: ignore[misc]


def test_records_reject_unknown_fields() -> None:
    with pytest.raises(PydanticValidationError):
        NodeStateRecord(run_id="run-1", node_id="a", provider="openai")  # type: ignore[call-arg]
