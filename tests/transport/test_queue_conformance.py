"""One contract, run against every ``WorkQueue`` implementation.

These are the properties Phase 8's correctness argument rests on, so they are asserted
rather than assumed:

* an item goes to exactly one consumer at a time (or two workers run the same node);
* a claimed item is *held*, not forgotten, until it is acknowledged (or a dead worker's
  node is lost forever);
* a stale claim can be taken over, keeping its receipt (or a dead worker's node is never
  run at all);
* a result is held until settled (or a coordinator crash loses the expansion it carried);
* a node definition survives the round trip (or a planner's fan-out arrives corrupted).

Every one of them is a property of *at-least-once delivery*, which is exactly what makes
the idempotency key from Phase 5 load-bearing rather than decorative.
"""

import pytest

from dagent.graph.builder import build_node
from dagent.models.state import NodeState
from dagent.transport.base import WorkItem, WorkQueue, WorkResult

pytestmark = pytest.mark.asyncio

FAST = 0.05
"""A short block. Long enough for a real Redis round trip, short enough not to be felt."""


async def claim_and_complete(
    queue: WorkQueue, item: WorkItem, *, consumer: str = "w1", **result: object
) -> WorkResult:
    """Submit, claim, and report one item — the ordinary life of a work item."""
    await queue.submit(item)
    claimed = await queue.claim(consumer=consumer, timeout_s=FAST)
    assert claimed is not None
    reported = WorkResult(
        claimed.run_id,
        claimed.node_id,
        claimed.attempt,
        NodeState.SUCCESS,
        **result,  # type: ignore[arg-type]
    )
    await queue.complete(claimed, reported)
    return reported


# --- the work channel -------------------------------------------------------------------


async def test_a_submitted_item_is_claimed_with_its_key_intact(queue: WorkQueue) -> None:
    await queue.submit(WorkItem("r1", "a", 3))

    claimed = await queue.claim(consumer="w1", timeout_s=FAST)

    assert claimed is not None
    assert (claimed.run_id, claimed.node_id, claimed.attempt) == ("r1", "a", 3)
    # The payload *is* the idempotency key: that is the whole reason redelivery is safe.
    assert claimed.idempotency_key == "r1:a:3"


async def test_claiming_an_empty_queue_returns_nothing_rather_than_waiting_forever(
    queue: WorkQueue,
) -> None:
    assert await queue.claim(consumer="w1", timeout_s=FAST) is None


async def test_an_item_claimed_by_one_worker_is_not_handed_to_another(queue: WorkQueue) -> None:
    await queue.submit(WorkItem("r1", "a", 0))

    first = await queue.claim(consumer="w1", timeout_s=FAST)
    second = await queue.claim(consumer="w2", timeout_s=FAST)

    assert first is not None
    assert second is None


async def test_two_workers_split_the_backlog_rather_than_duplicating_it(
    queue: WorkQueue,
) -> None:
    await queue.submit(WorkItem("r1", "a", 0))
    await queue.submit(WorkItem("r1", "b", 0))

    first = await queue.claim(consumer="w1", timeout_s=FAST)
    second = await queue.claim(consumer="w2", timeout_s=FAST)

    assert first is not None
    assert second is not None
    assert {first.node_id, second.node_id} == {"a", "b"}


async def test_every_delivery_gets_its_own_receipt(queue: WorkQueue) -> None:
    # Two deliveries of the same work carry the same key and different receipts. Without
    # that distinction, acknowledging one delivery would retire the other.
    await queue.submit(WorkItem("r1", "a", 0))
    await queue.submit(WorkItem("r1", "a", 0))

    first = await queue.claim(consumer="w1", timeout_s=FAST)
    second = await queue.claim(consumer="w1", timeout_s=FAST)

    assert first is not None
    assert second is not None
    assert first.idempotency_key == second.idempotency_key
    assert first.receipt != second.receipt


# --- redelivery, which is the whole point ------------------------------------------------


async def test_a_claim_nobody_acknowledged_can_be_taken_over(queue: WorkQueue) -> None:
    # This is a worker dying mid-node. Nothing about waiting makes the entry come back on
    # its own; another worker has to notice it has gone stale and claim it.
    await queue.submit(WorkItem("r1", "a", 2))
    claimed = await queue.claim(consumer="dead", timeout_s=FAST)
    assert claimed is not None

    reclaimed = await queue.reclaim(consumer="alive", min_idle_s=0)

    assert [(item.node_id, item.attempt) for item in reclaimed] == [("a", 2)]


async def test_a_reclaimed_item_keeps_the_attempt_it_was_dispatched_at(
    queue: WorkQueue,
) -> None:
    # The redelivered item names the *same* attempt, so the node comes back under the
    # idempotency key the outside world has already seen (DR-4). A fresh attempt number
    # here would silently turn one committed side effect into two.
    await queue.submit(WorkItem("r9", "charge", 7))
    claimed = await queue.claim(consumer="dead", timeout_s=FAST)
    assert claimed is not None

    (reclaimed,) = await queue.reclaim(consumer="alive", min_idle_s=0)

    assert reclaimed.idempotency_key == "r9:charge:7"


async def test_a_reclaimed_item_keeps_its_receipt(queue: WorkQueue) -> None:
    # Transferred, not reissued: whichever worker finishes first acknowledges the same
    # entry, and the loser's acknowledgement is a no-op instead of retiring a second copy.
    await queue.submit(WorkItem("r1", "a", 0))
    claimed = await queue.claim(consumer="dead", timeout_s=FAST)
    assert claimed is not None

    (reclaimed,) = await queue.reclaim(consumer="alive", min_idle_s=0)

    assert reclaimed.receipt == claimed.receipt


async def test_work_still_in_progress_is_not_stolen_from_a_living_worker(
    queue: WorkQueue,
) -> None:
    await queue.submit(WorkItem("r1", "a", 0))
    assert await queue.claim(consumer="busy", timeout_s=FAST) is not None

    assert await queue.reclaim(consumer="thief", min_idle_s=60) == ()


async def test_an_acknowledged_item_is_gone_for_good(queue: WorkQueue) -> None:
    await claim_and_complete(queue, WorkItem("r1", "a", 0))

    assert await queue.reclaim(consumer="w2", min_idle_s=0) == ()
    assert await queue.claim(consumer="w2", timeout_s=FAST) is None


async def test_nothing_to_reclaim_is_an_empty_answer_not_an_error(queue: WorkQueue) -> None:
    assert await queue.reclaim(consumer="w1", min_idle_s=0) == ()


# --- the results channel ----------------------------------------------------------------


async def test_a_completed_node_is_reported_to_the_coordinator(queue: WorkQueue) -> None:
    await queue.follow("r1")

    await claim_and_complete(queue, WorkItem("r1", "a", 1))
    collected = await queue.collect("r1", timeout_s=FAST)

    assert [(r.node_id, r.attempt, r.state) for r in collected] == [("a", 1, NodeState.SUCCESS)]


async def test_a_result_published_before_the_coordinator_looks_is_still_delivered(
    queue: WorkQueue,
) -> None:
    # A worker can be faster than the coordinator's first poll. A transport that dropped
    # what nobody was waiting for would hang the run on the node it lost.
    await queue.follow("r1")
    await claim_and_complete(queue, WorkItem("r1", "a", 0))

    assert len(await queue.collect("r1", timeout_s=FAST)) == 1


async def test_collecting_with_nothing_waiting_returns_nothing(queue: WorkQueue) -> None:
    await queue.follow("r1")

    assert await queue.collect("r1", timeout_s=FAST) == ()


async def test_results_are_only_ever_delivered_to_their_own_run(queue: WorkQueue) -> None:
    await queue.follow("r1")
    await queue.follow("r2")

    await claim_and_complete(queue, WorkItem("r2", "b", 0))

    assert await queue.collect("r1", timeout_s=FAST) == ()
    assert len(await queue.collect("r2", timeout_s=FAST)) == 1


async def test_a_result_is_held_until_it_is_settled(queue: WorkQueue) -> None:
    # The window this closes: a coordinator that read a planner's completion and died
    # before merging the expansion it carried. Acknowledging on receipt would lose it.
    await queue.follow("r1")
    await claim_and_complete(queue, WorkItem("r1", "a", 0))

    first = await queue.collect("r1", timeout_s=FAST)
    again = await queue.collect("r1", timeout_s=FAST)

    assert [r.receipt for r in first] == [r.receipt for r in again]


async def test_a_settled_result_is_not_delivered_again(queue: WorkQueue) -> None:
    await queue.follow("r1")
    await claim_and_complete(queue, WorkItem("r1", "a", 0))

    collected = await queue.collect("r1", timeout_s=FAST)
    await queue.settle(collected)

    assert await queue.collect("r1", timeout_s=FAST) == ()


async def test_settling_nothing_is_harmless(queue: WorkQueue) -> None:
    await queue.follow("r1")

    await queue.settle([])


async def test_following_a_run_twice_is_harmless(queue: WorkQueue) -> None:
    # `resume` does exactly this: a second coordinator follows a run the first one already
    # was, and must pick up whatever the first never acknowledged.
    await queue.follow("r1")
    await claim_and_complete(queue, WorkItem("r1", "a", 0))

    await queue.follow("r1")

    assert len(await queue.collect("r1", timeout_s=FAST)) == 1


# --- what rides on a result -------------------------------------------------------------


async def test_a_failed_node_reports_its_state(queue: WorkQueue) -> None:
    await queue.follow("r1")
    await queue.submit(WorkItem("r1", "a", 0))
    claimed = await queue.claim(consumer="w1", timeout_s=FAST)
    assert claimed is not None

    await queue.complete(claimed, WorkResult("r1", "a", 0, NodeState.FAILED))

    (collected,) = await queue.collect("r1", timeout_s=FAST)
    assert collected.state is NodeState.FAILED


async def test_an_expansion_survives_the_round_trip(queue: WorkQueue) -> None:
    # A planner's fan-out arrives at the coordinator as node *definitions*, because they
    # exist nowhere else yet — they are precisely the nodes not in the stored graph.
    await queue.follow("r1")
    grown = (
        build_node("plan.research_0", "researcher", depends_on=["plan"], params={"topic": "a"}),
        build_node("plan.synthesis", "synthesizer", inputs={"x": "plan.research_0"}),
    )

    await claim_and_complete(queue, WorkItem("r1", "plan", 0), expansion=grown)

    (collected,) = await queue.collect("r1", timeout_s=FAST)
    assert collected.expansion == grown


async def test_a_result_with_no_expansion_arrives_with_none(queue: WorkQueue) -> None:
    await queue.follow("r1")

    await claim_and_complete(queue, WorkItem("r1", "a", 0))

    (collected,) = await queue.collect("r1", timeout_s=FAST)
    assert collected.expansion == ()


async def test_every_implementation_satisfies_the_protocol(queue: WorkQueue) -> None:
    assert isinstance(queue, WorkQueue)
