"""Metering model calls against the run's budget — the admission half of FR-5.

Sits at the same seam as ``recording.py`` and composes with it: both are ordinary
``ModelClient`` implementations that wrap another one. The executor stacks them budget
outermost, so a refused call never reaches the recorder — there is no response to record,
and a replay must not find a call that never happened.

Lives in ``runtime`` rather than ``policy`` because it is the one piece of the budget story
that has to know about the model *seam*. The budget and its pricing know only about
``ModelResponse``, a plain schema, so ``dagent.policy`` keeps its one-way dependency on
``dagent.models`` and the package stays importable.
"""

from __future__ import annotations

from dagent.models.model_call import ModelRequest, ModelResponse
from dagent.policy.limits import Budget, Pricer, free
from dagent.runtime.model import ModelClient

__all__ = ["BudgetedModelClient"]


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
