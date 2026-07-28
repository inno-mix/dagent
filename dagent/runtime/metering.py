"""Metering model calls against the run's budget — the admission half of FR-5.

Sits at the same seam as ``recording.py`` and composes with it: both are ordinary
``ModelClient`` implementations that wrap another one. The executor stacks them budget
outermost, so a refused call never reaches the recorder — there is no response to record,
and a replay must not find a call that never happened.

Lives in ``runtime`` rather than ``policy`` because it is the one piece of the budget story
that has to know what a model call *is*. :class:`~dagent.policy.limits.Budget` itself holds
numbers only, so ``dagent.policy`` keeps its clean one-way dependency on ``dagent.models``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from dagent.models.model_call import ModelRequest, ModelResponse
from dagent.policy.limits import Budget
from dagent.runtime.model import ModelClient

__all__ = ["BudgetedModelClient", "Pricer", "free"]

Pricer: TypeAlias = Callable[[ModelResponse], float]
"""Turns a response into what it cost, in USD."""


def free(response: ModelResponse) -> float:
    """Price every call at zero — the default.

    No price table ships in this package on purpose. Per-token prices are vendor knowledge
    that changes without warning, and a stale table baked into an execution engine reports
    confident, wrong numbers. The token ceiling is exact and needs no table; a run that
    wants a dollar ceiling supplies the pricing it trusts.
    """
    return 0.0


class BudgetedModelClient:
    """A ``ModelClient`` that asks the budget for permission and charges it afterwards."""

    def __init__(
        self,
        inner: ModelClient,
        *,
        budget: Budget,
        price: Pricer = free,
    ) -> None:
        """Wrap ``inner``, metering every call it makes against ``budget``.

        Args:
            inner: The client that actually answers.
            budget: The run's shared budget. Shared, not per node — a ceiling that each
                node got its own copy of would not be a ceiling.
            price: Turns a response into a cost. Defaults to :func:`free`.
        """
        self._inner = inner
        self._budget = budget
        self._price = price

    @property
    def provider(self) -> str:
        """The wrapped client's provider — metering does not change who answers."""
        return self._inner.provider

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Admit, call, and charge.

        Raises:
            PolicyError: If the budget has already been spent. The call is not made, so
                "no further model calls are admitted" is enforced at the one place every
                model call passes through, rather than trusted to every agent.
        """
        self._budget.admit()
        response = await self._inner.complete(request)
        self._budget.charge(tokens=response.total_tokens, cost_usd=self._price(response))
        return response
