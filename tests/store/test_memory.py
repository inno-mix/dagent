import pytest

from dagent.errors import DagentError, StoreError
from dagent.models.state import NodeState, NodeStateRecord, RunState, RunStateRecord
from dagent.store.base import StateStore
from dagent.store.memory import InMemoryStateStore


def a_run(run_id: str = "r1") -> RunStateRecord:
    return RunStateRecord(run_id=run_id, workflow_name="w", state=RunState.RUNNING)


def test_the_memory_store_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryStateStore(), StateStore)


@pytest.mark.asyncio
async def test_a_checkpointed_run_can_be_loaded_back() -> None:
    store = InMemoryStateStore()

    await store.checkpoint(a_run())

    assert (await store.load_run("r1")).workflow_name == "w"


@pytest.mark.asyncio
async def test_checkpoint_replaces_the_stored_record() -> None:
    store = InMemoryStateStore()
    await store.checkpoint(a_run())

    await store.checkpoint(a_run().model_copy(update={"state": RunState.SUCCEEDED}))

    assert (await store.load_run("r1")).state is RunState.SUCCEEDED


@pytest.mark.asyncio
async def test_loading_an_unknown_run_raises() -> None:
    with pytest.raises(StoreError, match="unknown run"):
        await InMemoryStateStore().load_run("nope")


@pytest.mark.asyncio
async def test_node_state_is_stored_against_its_run() -> None:
    store = InMemoryStateStore()
    await store.checkpoint(a_run())

    await store.save_node_state(NodeStateRecord(run_id="r1", node_id="a", state=NodeState.RUNNING))

    assert (await store.load_run("r1")).nodes["a"].state is NodeState.RUNNING


@pytest.mark.asyncio
async def test_saving_node_state_replaces_the_previous_state() -> None:
    store = InMemoryStateStore()
    await store.checkpoint(a_run())

    await store.save_node_state(NodeStateRecord(run_id="r1", node_id="a", state=NodeState.RUNNING))
    await store.save_node_state(NodeStateRecord(run_id="r1", node_id="a", state=NodeState.SUCCESS))

    assert (await store.load_run("r1")).nodes["a"].state is NodeState.SUCCESS


@pytest.mark.asyncio
async def test_saving_node_state_leaves_sibling_nodes_alone() -> None:
    store = InMemoryStateStore()
    await store.checkpoint(a_run())

    await store.save_node_state(NodeStateRecord(run_id="r1", node_id="a", state=NodeState.SUCCESS))
    await store.save_node_state(NodeStateRecord(run_id="r1", node_id="b", state=NodeState.RUNNING))

    nodes = (await store.load_run("r1")).nodes
    assert nodes["a"].state is NodeState.SUCCESS
    assert nodes["b"].state is NodeState.RUNNING


@pytest.mark.asyncio
async def test_saving_node_state_for_an_unknown_run_raises() -> None:
    with pytest.raises(StoreError, match="unknown run"):
        await InMemoryStateStore().save_node_state(NodeStateRecord(run_id="ghost", node_id="a"))


@pytest.mark.asyncio
async def test_an_output_round_trips() -> None:
    store = InMemoryStateStore()
    await store.checkpoint(a_run())

    await store.append_output("r1", "a", {"summary": "hello"})

    assert await store.load_output("r1", "a") == {"summary": "hello"}


@pytest.mark.asyncio
async def test_the_latest_output_wins() -> None:
    store = InMemoryStateStore()
    await store.checkpoint(a_run())

    await store.append_output("r1", "a", {"attempt": 1})
    await store.append_output("r1", "a", {"attempt": 2})

    assert await store.load_output("r1", "a") == {"attempt": 2}


@pytest.mark.asyncio
async def test_outputs_are_scoped_to_their_run() -> None:
    store = InMemoryStateStore()
    await store.checkpoint(a_run("r1"))
    await store.checkpoint(a_run("r2"))

    await store.append_output("r1", "a", "first")
    await store.append_output("r2", "a", "second")

    assert await store.load_output("r1", "a") == "first"
    assert await store.load_output("r2", "a") == "second"


@pytest.mark.asyncio
async def test_appending_an_output_for_an_unknown_run_raises() -> None:
    with pytest.raises(StoreError, match="unknown run"):
        await InMemoryStateStore().append_output("ghost", "a", None)


@pytest.mark.asyncio
async def test_loading_a_missing_output_raises_rather_than_returning_none() -> None:
    # None is a legitimate output, so absence has to be signalled out of band.
    store = InMemoryStateStore()
    await store.checkpoint(a_run())

    with pytest.raises(StoreError, match="no output"):
        await store.load_output("r1", "a")


@pytest.mark.asyncio
async def test_a_stored_none_output_is_distinguishable_from_a_missing_one() -> None:
    store = InMemoryStateStore()
    await store.checkpoint(a_run())

    await store.append_output("r1", "a", None)

    assert await store.load_output("r1", "a") is None


@pytest.mark.asyncio
async def test_store_failures_are_catchable_as_dagent_errors() -> None:
    with pytest.raises(DagentError):
        await InMemoryStateStore().load_run("nope")


@pytest.mark.asyncio
async def test_two_stores_share_nothing() -> None:
    first, second = InMemoryStateStore(), InMemoryStateStore()
    await first.checkpoint(a_run())

    with pytest.raises(StoreError):
        await second.load_run("r1")
