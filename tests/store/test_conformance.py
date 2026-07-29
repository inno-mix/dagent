"""One contract, every implementation.

DR-3 claims the executor is storage-agnostic. That claim is only worth anything if both
stores actually behave the same, so this file states the `StateStore` contract once and
runs it against each of them. A behaviour that is not asserted here is a behaviour the
executor must not rely on.
"""

import pytest

from dagent.errors import StoreError
from dagent.models.model_call import ModelCallRecord, ModelRequest, ModelResponse
from dagent.models.state import NodeOutput, NodeState, NodeStateRecord, RunState, RunStateRecord
from dagent.models.workflow import Node, Policy, Workflow
from dagent.store.base import StateStore

WORKFLOW = Workflow(
    name="conformance",
    nodes=(
        Node(id="a", agent="fake", params={"value": "seed"}),
        Node(
            id="b",
            agent="fake",
            depends_on=("a",),
            inputs={"x": "a"},
            policy=Policy(max_attempts=3, timeout_s=2.5),
        ),
    ),
)


def a_run(run_id: str = "r1", state: RunState = RunState.RUNNING) -> RunStateRecord:
    return RunStateRecord(run_id=run_id, workflow_name="conformance", state=state)


async def opened(store: StateStore, run_id: str = "r1") -> str:
    """Open a run the way the executor does: definition first, then the record."""
    await store.save_workflow(run_id, WORKFLOW)
    await store.checkpoint(a_run(run_id))
    return run_id


@pytest.mark.asyncio
async def test_every_store_satisfies_the_protocol(store: StateStore) -> None:
    assert isinstance(store, StateStore)


# --- runs -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_checkpointed_run_reads_back(store: StateStore) -> None:
    await opened(store)

    run = await store.load_run("r1")

    assert (run.run_id, run.workflow_name, run.state) == ("r1", "conformance", RunState.RUNNING)


@pytest.mark.asyncio
async def test_an_unknown_run_raises_rather_than_returning_none(store: StateStore) -> None:
    # `None` is a legal output, so absence has to be signalled out of band.
    with pytest.raises(StoreError, match="unknown run"):
        await store.load_run("never-existed")


@pytest.mark.asyncio
async def test_checkpointing_again_replaces_the_record(store: StateStore) -> None:
    await opened(store)

    await store.checkpoint(a_run(state=RunState.SUCCEEDED))

    assert (await store.load_run("r1")).state is RunState.SUCCEEDED


@pytest.mark.asyncio
async def test_runs_do_not_leak_into_one_another(store: StateStore) -> None:
    await opened(store, "first")
    await opened(store, "second")

    await store.save_node_state(
        NodeStateRecord(run_id="first", node_id="a", state=NodeState.SUCCESS)
    )

    assert (await store.load_run("second")).nodes == {}


# --- definitions ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stored_workflow_reads_back_identically(store: StateStore) -> None:
    # Byte-identical, not merely similar: resume executes what comes back out of here, so
    # a lossy round trip is a run that continues as a different workflow.
    await opened(store)

    assert await store.load_workflow("r1") == WORKFLOW


@pytest.mark.asyncio
async def test_a_stored_workflow_keeps_its_node_policies(store: StateStore) -> None:
    await opened(store)

    restored = await store.load_workflow("r1")

    assert restored.nodes[1].policy == Policy(max_attempts=3, timeout_s=2.5)


@pytest.mark.asyncio
async def test_a_run_without_a_stored_workflow_says_so(store: StateStore) -> None:
    await store.checkpoint(a_run("bare"))

    with pytest.raises(StoreError, match="no stored workflow"):
        await store.load_workflow("bare")


# --- node states ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_node_state_reads_back_through_the_run(store: StateStore) -> None:
    await opened(store)

    await store.save_node_state(
        NodeStateRecord(run_id="r1", node_id="a", state=NodeState.FAILED, attempt=2, error="boom")
    )

    record = (await store.load_run("r1")).nodes["a"]
    assert (record.state, record.attempt, record.error) == (NodeState.FAILED, 2, "boom")


@pytest.mark.asyncio
async def test_saving_a_node_state_replaces_the_previous_one(store: StateStore) -> None:
    # The store holds current state, not a history: an attempt that is retried overwrites.
    await opened(store)

    await store.save_node_state(NodeStateRecord(run_id="r1", node_id="a", state=NodeState.RUNNING))
    await store.save_node_state(
        NodeStateRecord(run_id="r1", node_id="a", state=NodeState.SUCCESS, attempt=1)
    )

    nodes = (await store.load_run("r1")).nodes
    assert len(nodes) == 1
    assert (nodes["a"].state, nodes["a"].attempt) == (NodeState.SUCCESS, 1)


@pytest.mark.asyncio
async def test_a_checkpoint_carries_its_node_map(store: StateStore) -> None:
    await store.save_workflow("r1", WORKFLOW)
    await store.checkpoint(
        RunStateRecord(
            run_id="r1",
            workflow_name="conformance",
            nodes={"a": NodeStateRecord(run_id="r1", node_id="a", state=NodeState.SUCCESS)},
        )
    )

    assert (await store.load_run("r1")).nodes["a"].state is NodeState.SUCCESS


# --- outputs --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output",
    [
        "a string",
        42,
        3.5,
        True,
        None,
        {"nested": {"list": [1, 2, {"deep": None}]}},
        [],
        {},
    ],
    ids=["str", "int", "float", "bool", "none", "nested", "empty-list", "empty-dict"],
)
@pytest.mark.asyncio
async def test_an_output_round_trips_unchanged(store: StateStore, output: NodeOutput) -> None:
    await opened(store)

    await store.append_output("r1", "a", output)

    assert await store.load_output("r1", "a") == output


@pytest.mark.asyncio
async def test_a_none_output_is_distinguishable_from_a_missing_one(store: StateStore) -> None:
    # The reason `load_output` raises instead of returning None.
    await opened(store)
    await store.append_output("r1", "a", None)

    assert await store.load_output("r1", "a") is None
    with pytest.raises(StoreError, match="no output"):
        await store.load_output("r1", "b")


@pytest.mark.asyncio
async def test_writing_an_output_twice_is_safe(store: StateStore) -> None:
    # A resumed node re-writes the output it may already have written. Every write in
    # this store has to survive being repeated (DR-4).
    await opened(store)

    await store.append_output("r1", "a", {"take": 1})
    await store.append_output("r1", "a", {"take": 2})

    assert await store.load_output("r1", "a") == {"take": 2}


# --- model calls ----------------------------------------------------------------------


def a_call(node_id: str, attempt: int = 0, sequence: int = 0) -> ModelCallRecord:
    return ModelCallRecord(
        run_id="r1",
        node_id=node_id,
        attempt=attempt,
        sequence=sequence,
        request=ModelRequest(prompt=f"{node_id}/{attempt}/{sequence}"),
        response=ModelResponse(
            text="ok", provider="stub", model="m", input_tokens=3, output_tokens=4
        ),
    )


@pytest.mark.asyncio
async def test_model_calls_read_back(store: StateStore) -> None:
    await opened(store)

    await store.append_model_call(a_call("a"))
    await store.append_model_call(a_call("b"))

    calls = await store.load_model_calls("r1")
    assert {call.node_id for call in calls} == {"a", "b"}
    assert calls[0].response.total_tokens == 7


@pytest.mark.asyncio
async def test_a_run_with_no_model_calls_returns_nothing_rather_than_raising(
    store: StateStore,
) -> None:
    await opened(store)

    assert list(await store.load_model_calls("r1")) == []


@pytest.mark.asyncio
async def test_model_calls_for_an_unknown_run_raise(store: StateStore) -> None:
    with pytest.raises(StoreError, match="unknown run"):
        await store.load_model_calls("never-existed")


@pytest.mark.asyncio
async def test_a_node_may_record_several_calls(store: StateStore) -> None:
    await opened(store)

    await store.append_model_call(a_call("a", sequence=0))
    await store.append_model_call(a_call("a", sequence=1))

    assert len(await store.load_model_calls("r1")) == 2


@pytest.mark.asyncio
async def test_calls_from_different_attempts_are_both_kept(store: StateStore) -> None:
    # (run_id, node_id, attempt, sequence) is what a replay looks calls up by, so a retry
    # must not overwrite the record of the attempt before it.
    await opened(store)

    await store.append_model_call(a_call("a", attempt=0))
    await store.append_model_call(a_call("a", attempt=1))

    assert {call.attempt for call in await store.load_model_calls("r1")} == {0, 1}


@pytest.mark.asyncio
async def test_recording_the_same_call_twice_does_not_duplicate_it(store: StateStore) -> None:
    # A re-executed attempt replays its own sequence numbers. Two copies of one call would
    # be two copies in the replay.
    await opened(store)

    await store.append_model_call(a_call("a"))
    await store.append_model_call(a_call("a"))

    assert len(await store.load_model_calls("r1")) == 1


@pytest.mark.asyncio
async def test_an_output_for_an_unknown_run_raises(store: StateStore) -> None:
    with pytest.raises(StoreError):
        await store.load_output("never-existed", "a")
