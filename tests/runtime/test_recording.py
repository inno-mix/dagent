import pytest

from dagent.errors import AgentError
from dagent.models.model_call import ModelRequest
from dagent.models.state import RunState, RunStateRecord
from dagent.runtime.clock import ManualClock
from dagent.runtime.model import ModelClient, NullModelClient, StubModelClient
from dagent.runtime.recording import RecordingModelClient
from dagent.store.memory import InMemoryStateStore


async def a_store(run_id: str = "r1") -> InMemoryStateStore:
    store = InMemoryStateStore()
    await store.checkpoint(RunStateRecord(run_id=run_id, workflow_name="w", state=RunState.RUNNING))
    return store


def test_the_wrapper_satisfies_the_protocol() -> None:
    wrapper = RecordingModelClient(
        NullModelClient(), store=InMemoryStateStore(), run_id="r", node_id="n"
    )

    assert isinstance(wrapper, ModelClient)


@pytest.mark.asyncio
async def test_the_wrapper_reports_the_wrapped_provider() -> None:
    store = await a_store()
    wrapper = RecordingModelClient(
        StubModelClient(["x"], provider="gemini"), store=store, run_id="r1", node_id="n"
    )

    assert wrapper.provider == "gemini"


@pytest.mark.asyncio
async def test_the_response_passes_through_unchanged() -> None:
    store = await a_store()
    wrapper = RecordingModelClient(
        StubModelClient(["the answer"]), store=store, run_id="r1", node_id="n"
    )

    response = await wrapper.complete(ModelRequest(prompt="q"))

    assert response.text == "the answer"


@pytest.mark.asyncio
async def test_each_call_is_written_into_the_run_record() -> None:
    store = await a_store()
    wrapper = RecordingModelClient(
        StubModelClient(["a", "b"]), store=store, run_id="r1", node_id="n", attempt=2
    )

    await wrapper.complete(ModelRequest(prompt="first"))
    await wrapper.complete(ModelRequest(prompt="second"))

    calls = await store.load_model_calls("r1")
    assert [call.request.prompt for call in calls] == ["first", "second"]
    assert [call.response.text for call in calls] == ["a", "b"]
    assert {call.node_id for call in calls} == {"n"}
    assert {call.attempt for call in calls} == {2}


@pytest.mark.asyncio
async def test_calls_are_numbered_so_a_replay_can_match_them_up() -> None:
    # (run_id, node_id, attempt, sequence) is the key a replay client will look up.
    store = await a_store()
    wrapper = RecordingModelClient(
        StubModelClient(["a", "b", "c"]), store=store, run_id="r1", node_id="n"
    )

    for prompt in ("1", "2", "3"):
        await wrapper.complete(ModelRequest(prompt=prompt))

    assert [call.sequence for call in await store.load_model_calls("r1")] == [0, 1, 2]


@pytest.mark.asyncio
async def test_two_nodes_record_independently_into_the_same_run() -> None:
    store = await a_store()
    first = RecordingModelClient(StubModelClient(["a"]), store=store, run_id="r1", node_id="one")
    second = RecordingModelClient(StubModelClient(["b"]), store=store, run_id="r1", node_id="two")

    await first.complete(ModelRequest(prompt="x"))
    await second.complete(ModelRequest(prompt="y"))

    calls = await store.load_model_calls("r1")
    assert {(call.node_id, call.sequence) for call in calls} == {("one", 0), ("two", 0)}


@pytest.mark.asyncio
async def test_timestamps_come_from_the_injected_clock() -> None:
    store = await a_store()
    clock = ManualClock()
    wrapper = RecordingModelClient(
        StubModelClient(["a"]), store=store, run_id="r1", node_id="n", clock=clock
    )

    await wrapper.complete(ModelRequest(prompt="x"))

    (call,) = await store.load_model_calls("r1")
    assert call.started_at == clock.now()
    assert call.finished_at == clock.now()


@pytest.mark.asyncio
async def test_a_failed_call_records_nothing() -> None:
    # There is no response to replay, and Phase 4's retry will make its own call.
    store = await a_store()
    wrapper = RecordingModelClient(NullModelClient(), store=store, run_id="r1", node_id="n")

    with pytest.raises(AgentError):
        await wrapper.complete(ModelRequest(prompt="x"))

    assert await store.load_model_calls("r1") == ()
