import pytest

from dagent.errors import AgentError, DagentError
from dagent.models.model_call import ModelRequest, ModelResponse
from dagent.runtime.model import ModelClient, NullModelClient, StubModelClient


def test_the_null_client_satisfies_the_protocol() -> None:
    assert isinstance(NullModelClient(), ModelClient)


def test_the_stub_client_satisfies_the_protocol() -> None:
    assert isinstance(StubModelClient(), ModelClient)


def test_the_null_client_reports_a_provider_name() -> None:
    # Phase 4 keys per-provider caps on this, so it must never be absent.
    assert NullModelClient().provider == "null"


@pytest.mark.asyncio
async def test_the_null_client_explains_what_is_missing_rather_than_crashing() -> None:
    with pytest.raises(AgentError, match="no model client"):
        await NullModelClient().complete(ModelRequest(prompt="hi"))


@pytest.mark.asyncio
async def test_the_null_client_failure_is_catchable_as_a_dagent_error() -> None:
    with pytest.raises(DagentError):
        await NullModelClient().complete(ModelRequest(prompt="hi"))


@pytest.mark.asyncio
async def test_the_stub_returns_its_scripted_replies_in_order() -> None:
    stub = StubModelClient(["first", "second"])

    assert (await stub.complete(ModelRequest(prompt="a"))).text == "first"
    assert (await stub.complete(ModelRequest(prompt="b"))).text == "second"


@pytest.mark.asyncio
async def test_the_stub_records_every_request_it_saw() -> None:
    stub = StubModelClient(["x", "y"])

    await stub.complete(ModelRequest(prompt="one"))
    await stub.complete(ModelRequest(prompt="two"))

    assert [request.prompt for request in stub.requests] == ["one", "two"]


@pytest.mark.asyncio
async def test_the_stub_refuses_to_run_past_its_script() -> None:
    # Repeating the last reply would let a test pass while the agent made more calls
    # than it was supposed to.
    stub = StubModelClient(["only one"])
    await stub.complete(ModelRequest(prompt="a"))

    with pytest.raises(AgentError, match="scripted with 1 replies"):
        await stub.complete(ModelRequest(prompt="b"))


@pytest.mark.asyncio
async def test_the_stub_can_answer_as_a_function_of_the_request() -> None:
    stub = StubModelClient(lambda request: request.prompt.upper())

    assert (await stub.complete(ModelRequest(prompt="shout"))).text == "SHOUT"


@pytest.mark.asyncio
async def test_a_functional_stub_never_runs_out() -> None:
    stub = StubModelClient(lambda request: "always")

    for _ in range(5):
        await stub.complete(ModelRequest(prompt="hi"))

    assert len(stub.requests) == 5


@pytest.mark.asyncio
async def test_the_stub_reports_its_configured_provider_and_model() -> None:
    stub = StubModelClient(["x"], provider="pretend", model="pretend-1")

    response = await stub.complete(ModelRequest(prompt="hi"))

    assert stub.provider == "pretend"
    assert (response.provider, response.model) == ("pretend", "pretend-1")


@pytest.mark.asyncio
async def test_a_request_level_model_wins_over_the_stub_default() -> None:
    stub = StubModelClient(["x"])

    response = await stub.complete(ModelRequest(prompt="hi", model="explicit"))

    assert response.model == "explicit"


@pytest.mark.asyncio
async def test_the_stub_reports_plausible_token_usage() -> None:
    # Phase 4's budget meters on these, so they must not be zero by default.
    stub = StubModelClient(["one two three"])

    response = await stub.complete(ModelRequest(prompt="a b"))

    assert response.input_tokens == 2
    assert response.output_tokens == 3
    assert response.total_tokens == 5


def test_total_tokens_sums_both_directions() -> None:
    response = ModelResponse(text="x", provider="p", model="m", input_tokens=7, output_tokens=3)

    assert response.total_tokens == 10
