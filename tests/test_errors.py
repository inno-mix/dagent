import pytest

from dagent.errors import (
    AgentError,
    DagentError,
    PolicyError,
    StoreError,
    ValidationError,
)

LEAF_ERRORS = [ValidationError, PolicyError, AgentError, StoreError]


def test_dagent_error_is_an_exception() -> None:
    assert issubclass(DagentError, Exception)


@pytest.mark.parametrize("error_type", LEAF_ERRORS, ids=lambda t: t.__name__)
def test_every_error_derives_from_dagent_error(error_type: type[DagentError]) -> None:
    # AGENTS.md §5: one catchable root, so a caller can tell engine failures from
    # dependency failures without enumerating types.
    assert issubclass(error_type, DagentError)


@pytest.mark.parametrize("error_type", LEAF_ERRORS, ids=lambda t: t.__name__)
def test_errors_carry_their_message(error_type: type[DagentError]) -> None:
    assert str(error_type("something went wrong")) == "something went wrong"


def test_validation_error_without_a_cycle_reports_none() -> None:
    assert ValidationError("no upstream produces 'x'").cycle is None


def test_validation_error_normalises_its_cycle_to_a_tuple() -> None:
    error = ValidationError("cycle: a -> b -> a", cycle=["a", "b", "a"])

    assert error.cycle == ("a", "b", "a")
    assert str(error) == "cycle: a -> b -> a"


def test_validation_error_cycle_is_immutable() -> None:
    path = ["a", "b", "a"]
    error = ValidationError("cycle", cycle=path)
    path.append("c")

    assert error.cycle == ("a", "b", "a")
