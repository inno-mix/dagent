"""Run-level policy: failure semantics, and the bundle of knobs the executor is handed.

Node-level policy (attempts, backoff, timeout) belongs to the *definition* and lives in
:class:`~dagent.models.workflow.Policy`. Everything here belongs to the *run*: it is how
this execution of the workflow behaves, not what the workflow is, so it is supplied at
submit time and never frozen into the definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from dagent.models.workflow import Policy
from dagent.policy.limits import Budget, Limits, Pricer, free
from dagent.policy.retry import Backoff, Retryable, default_retryable

__all__ = ["FailureMode", "RunPolicy"]


class FailureMode(StrEnum):
    """What a node's failure does to the rest of the run (FR-5).

    Three modes because the right answer genuinely depends on the workflow, and a
    scheduler that picks one on its own is making a product decision it has no business
    making. A batch of independent summaries wants every branch to finish; a pipeline
    feeding a paid downstream step wants to stop the moment anything is wrong.
    """

    RUN_TO_COMPLETION = "run_to_completion"
    """Let every independent branch finish. Nodes blocked by the failure stay ``PENDING``."""

    FAIL_FAST = "fail_fast"
    """Cancel everything still in flight and dispatch nothing further."""

    SKIP_DOWNSTREAM = "skip_downstream"
    """Finish independent branches, and mark the failure's descendants ``SKIPPED``.

    Same execution as ``run_to_completion``; the difference is honesty in the record.
    A node that can never become ready is reported as skipped rather than left looking
    like work that is still to come.
    """


@dataclass(frozen=True, slots=True)
class RunPolicy:
    """Everything the executor needs to know about limits, retries, and failure.

    Every default is deliberately inert: no cap, no ceiling, one attempt per node, no
    timeout, independent branches allowed to finish. A run that passes no policy behaves
    exactly as it did before the policy layer existed, so opting in to a limit is a
    visible decision rather than an inherited surprise.

    Frozen, but it holds two mutable objects — ``limits`` and ``budget`` accumulate state
    across a run by design. Freezing the bundle stops the *configuration* changing
    mid-run; the counters inside it are the run's own bookkeeping.
    """

    failure_mode: FailureMode = FailureMode.RUN_TO_COMPLETION
    node_defaults: Policy = field(default_factory=Policy)
    """Applied to any node that declares no ``policy`` of its own."""
    limits: Limits = field(default_factory=Limits)
    budget: Budget = field(default_factory=Budget)
    backoff: Backoff = field(default_factory=Backoff)
    max_expansion_depth: int = 1
    """How many generations of dynamic expansion a run allows (FR-7).

    One by default: a planner may grow the graph, and what it grows may not grow further.
    That covers the showcase workflow and stops an agent that emits a copy of itself from
    running until the budget or the node ceiling notices. Set 0 to forbid expansion
    outright, or higher for a planner of planners.

    Counted in memory, per run: a resumed run treats everything already in the stored
    graph as generation zero, so depth alone does not bound a run across many crashes.
    :attr:`max_graph_nodes` is the bound that does, because it is measured against the
    persisted graph.
    """
    max_graph_nodes: int = 1000
    """A ceiling on the size of the graph however it got that big.

    Not inert, unlike the other limits, because this one is a runaway guard rather than a
    policy choice — and 1000 nodes is already an order of magnitude past what SPEC's
    performance target contemplates for v1.
    """
    price: Pricer = field(default=free)
    """Turns a model response into a cost, for the budget's dollar ceiling. Defaults to
    free, because no price table ships with the engine — see :func:`~dagent.policy.limits.free`.
    A resumed run re-prices its recorded calls through this, so the ceiling survives a crash."""
    retryable: Retryable = field(default=default_retryable)
    """Decides whether a failed attempt earns another one. Swap it to change the rules
    without touching the executor."""

    def policy_for(self, override: Policy | None) -> Policy:
        """Return the policy governing a node: its own override, or the run's default."""
        return override if override is not None else self.node_defaults
