"""The budget wrapper: admission before the call, charging after it."""

import pytest

from dagent.errors import PolicyError
from dagent.models.model_call import ModelRequest, ModelResponse
from dagent.policy.limits import Budget
from dagent.runtime.metering import BudgetedModelClient, free
from dagent.runtime.model import ModelClient, StubModelClient


def metered(
    budget: Budget, replies: tuple[str, ...] = ("hello there",)
) -> tuple[BudgetedModelClient, StubModelClient]:
    inner = StubModelClient(replies)
    return BudgetedModelClient(inner, budget=budget), inner


def test_the_wrapper_satisfies_the_model_client_protocol() -> None:
    assert isinstance(BudgetedModelClient(StubModelClient(), budget=Budget()), ModelClient)


def test_the_provider_name_passes_through_unchanged() -> None:
    # Per-provider caps key on this, so metering must not rename who answers.
    inner = StubModelClient(provider="gemini")

    assert BudgetedModelClient(inner, budget=Budget()).provider == "gemini"


@pytest.mark.asyncio
async def test_a_completed_call_charges_its_tokens_to_the_budget() -> None:
    budget = Budget()
    client, _ = metered(budget)

    response = await client.complete(ModelRequest(prompt="one two three"))

    assert budget.tokens_used == response.total_tokens
    assert budget.tokens_used > 0


@pytest.mark.asyncio
async def test_charges_accumulate_across_calls() -> None:
    budget = Budget()
    client, _ = metered(budget, replies=("a b", "c d e"))

    first = await client.complete(ModelRequest(prompt="x"))
    second = await client.complete(ModelRequest(prompt="y"))

    assert budget.tokens_used == first.total_tokens + second.total_tokens


@pytest.mark.asyncio
async def test_a_call_past_the_ceiling_is_refused() -> None:
    budget = Budget(max_tokens=1)
    client, _ = metered(budget, replies=("a b c", "should never be sent"))

    await client.complete(ModelRequest(prompt="first"))

    with pytest.raises(PolicyError, match="no further model calls"):
        await client.complete(ModelRequest(prompt="second"))


@pytest.mark.asyncio
async def test_a_refused_call_never_reaches_the_provider() -> None:
    # "No further model calls are admitted" has to mean the socket, not just the ledger.
    budget = Budget(max_tokens=1)
    client, inner = metered(budget, replies=("a b c",))

    await client.complete(ModelRequest(prompt="first"))
    with pytest.raises(PolicyError):
        await client.complete(ModelRequest(prompt="second"))

    assert [request.prompt for request in inner.requests] == ["first"]


@pytest.mark.asyncio
async def test_a_failed_call_is_not_charged() -> None:
    # There is nothing to pay for: the provider never produced anything.
    class Broken:
        @property
        def provider(self) -> str:
            return "broken"

        async def complete(self, request: ModelRequest) -> ModelResponse:
            raise RuntimeError("connection reset")

    budget = Budget()
    client = BudgetedModelClient(Broken(), budget=budget)

    with pytest.raises(RuntimeError):
        await client.complete(ModelRequest(prompt="hi"))

    assert budget.tokens_used == 0


@pytest.mark.asyncio
async def test_cost_comes_from_an_injected_pricer() -> None:
    budget = Budget()
    client = BudgetedModelClient(
        StubModelClient(("a",)),
        budget=budget,
        price=lambda response: response.total_tokens * 0.5,
    )

    response = await client.complete(ModelRequest(prompt="hi"))

    assert budget.cost_used == response.total_tokens * 0.5


@pytest.mark.asyncio
async def test_calls_are_free_by_default() -> None:
    # No price table ships in the engine: a stale one reports confident, wrong numbers.
    budget = Budget()
    client, _ = metered(budget)

    await client.complete(ModelRequest(prompt="hi"))

    assert budget.cost_used == 0.0


def test_the_default_pricer_prices_everything_at_zero() -> None:
    response = ModelResponse(text="x", provider="p", model="m", input_tokens=99)

    assert free(response) == 0.0


@pytest.mark.asyncio
async def test_one_budget_shared_by_two_clients_is_still_one_ceiling() -> None:
    # Each node gets its own wrapper; a per-node ceiling would not be a run ceiling.
    budget = Budget(max_tokens=100)
    first, _ = metered(budget, replies=("a b c",))
    second, _ = metered(budget, replies=("d e f",))

    await first.complete(ModelRequest(prompt="one"))
    await second.complete(ModelRequest(prompt="two"))

    assert budget.tokens_used > 0
