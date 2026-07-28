"""The run budget: what it counts, when it refuses, and the refused/exceeded distinction."""

import pytest

from dagent.errors import PolicyError
from dagent.policy.limits import Budget


def test_an_unbounded_budget_never_refuses() -> None:
    budget = Budget()

    budget.charge(tokens=10_000_000, cost_usd=999.0)
    budget.admit()

    assert budget.exceeded is False
    assert budget.refused is False


def test_charging_accumulates() -> None:
    budget = Budget()

    budget.charge(tokens=30)
    budget.charge(tokens=12, cost_usd=0.25)

    assert (budget.tokens_used, budget.cost_used) == (42, 0.25)


def test_a_call_is_admitted_while_under_the_token_ceiling() -> None:
    budget = Budget(max_tokens=100)
    budget.charge(tokens=99)

    budget.admit()

    assert budget.refused is False


def test_reaching_the_token_ceiling_refuses_the_next_call() -> None:
    # Reached, not passed: a budget that admits one more call at exactly the ceiling is a
    # ceiling that can always be overshot by one completion of unknown size.
    budget = Budget(max_tokens=100)
    budget.charge(tokens=100)

    with pytest.raises(PolicyError, match="budget exceeded"):
        budget.admit()


def test_the_cost_ceiling_refuses_independently_of_tokens() -> None:
    budget = Budget(max_cost_usd=1.0)
    budget.charge(tokens=3, cost_usd=1.5)

    with pytest.raises(PolicyError, match=r"\$1.5000/\$1.0000"):
        budget.admit()


def test_every_refusal_is_counted() -> None:
    budget = Budget(max_tokens=1)
    budget.charge(tokens=5)

    for _ in range(3):
        with pytest.raises(PolicyError):
            budget.admit()

    assert budget.refusals == 3


def test_a_ceiling_crossed_by_the_last_call_is_exceeded_but_not_refused() -> None:
    # The distinction that decides the run's outcome: nothing was stopped, so this run
    # succeeded. Reporting BUDGET_EXCEEDED here would turn a complete run into a failure.
    budget = Budget(max_tokens=10)
    budget.charge(tokens=50)

    assert budget.exceeded is True
    assert budget.refused is False


def test_refusal_is_what_marks_a_budget_as_having_stopped_work() -> None:
    budget = Budget(max_tokens=10)
    budget.charge(tokens=50)

    with pytest.raises(PolicyError):
        budget.admit()

    assert budget.refused is True


def test_describe_shows_usage_against_an_infinite_ceiling() -> None:
    budget = Budget()
    budget.charge(tokens=7)

    assert budget.describe() == "tokens 7/∞"


def test_describe_shows_both_ceilings_when_both_are_set() -> None:
    budget = Budget(max_tokens=100, max_cost_usd=2.0)
    budget.charge(tokens=40, cost_usd=0.5)

    assert budget.describe() == "tokens 40/100, cost $0.5000/$2.0000"


@pytest.mark.parametrize(
    "kwargs", [{"max_tokens": -1}, {"max_cost_usd": -0.01}], ids=["tokens", "cost"]
)
def test_a_negative_ceiling_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(PolicyError, match="must not be negative"):
        Budget(**kwargs)  # type: ignore[arg-type]


def test_a_zero_ceiling_refuses_everything_from_the_start() -> None:
    # Legal, and occasionally what you want: a dry run that admits no model call at all.
    with pytest.raises(PolicyError):
        Budget(max_tokens=0).admit()
