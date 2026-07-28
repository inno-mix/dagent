"""RunPolicy: the defaults must be inert, and node overrides must win."""

from dagent.models.workflow import Policy
from dagent.policy.limits import Budget, Limits
from dagent.policy.retry import default_retryable, full_jitter
from dagent.policy.run import FailureMode, RunPolicy


def test_the_three_failure_modes_spec_5_names_all_exist() -> None:
    expected = {"run_to_completion", "fail_fast", "skip_downstream"}

    assert {mode.value for mode in FailureMode} == expected


def test_the_default_failure_mode_lets_independent_branches_finish() -> None:
    # The least surprising of the three, and the one Phase 2 already behaved like.
    assert RunPolicy().failure_mode is FailureMode.RUN_TO_COMPLETION


def test_every_default_is_inert() -> None:
    # A run that passes no policy must behave exactly as it did with no policy layer:
    # one attempt, no timeout, no cap, no ceiling.
    policy = RunPolicy()

    assert policy.node_defaults == Policy()
    assert policy.node_defaults.max_attempts == 1
    assert policy.node_defaults.timeout_s is None
    assert policy.budget.exceeded is False
    assert policy.limits.peak("anything") == 0
    assert policy.retryable is default_retryable
    assert policy.backoff.jitter is full_jitter


def test_each_run_policy_gets_its_own_limits_and_budget() -> None:
    # Shared mutable defaults across runs would leak one run's spend into the next.
    first, second = RunPolicy(), RunPolicy()

    first.budget.charge(tokens=100)

    assert second.budget.tokens_used == 0
    assert first.limits is not second.limits


def test_a_nodes_own_policy_wins_over_the_run_default() -> None:
    policy = RunPolicy(node_defaults=Policy(max_attempts=5))
    override = Policy(max_attempts=2, timeout_s=1.0)

    assert policy.policy_for(override) is override


def test_a_node_without_a_policy_inherits_the_run_default() -> None:
    defaults = Policy(max_attempts=5)

    assert RunPolicy(node_defaults=defaults).policy_for(None) is defaults


def test_the_configuration_cannot_be_swapped_mid_run() -> None:
    # Frozen: the counters inside change, the rules do not.
    policy = RunPolicy()

    try:
        policy.failure_mode = FailureMode.FAIL_FAST  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("RunPolicy should be frozen")


def test_limits_and_budget_are_accepted_by_injection() -> None:
    limits, budget = Limits(max_concurrency=2), Budget(max_tokens=10)
    policy = RunPolicy(limits=limits, budget=budget)

    assert (policy.limits, policy.budget) == (limits, budget)
