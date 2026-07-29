"""Agents that do no I/O — for examples, and for exercising the engine without a model.

Shipped rather than confined to the test suite because anyone building a workflow needs
to run its shape before its prompts exist, and because every test in this repository is
required to be network-free (AGENTS.md §6).
"""

from __future__ import annotations

import asyncio

from dagent.errors import AgentError
from dagent.models.state import NodeOutput
from dagent.runtime.agent import AgentContext
from dagent.runtime.registry import register

__all__ = [
    "ConstantAgent",
    "EchoAgent",
    "FailingAgent",
    "FakeAgent",
    "FlakyAgent",
    "HangingAgent",
    "SideEffectAgent",
]


@register("fake")
class FakeAgent:
    """Returns a deterministic record of what it was given.

    Deterministic on purpose: two runs of the same workflow produce byte-identical
    output, which is what lets Phase 5 compare an interrupted run against an
    uninterrupted one.
    """

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Echo the node's identity and its resolved inputs."""
        return {
            "node_id": ctx.node_id,
            "attempt": ctx.attempt,
            # Sorted so the output does not depend on the order inputs were declared.
            "inputs": dict(sorted(ctx.inputs.items())),
        }


@register("constant")
class ConstantAgent:
    """Emits its ``value`` parameter unchanged — how a workflow gets its seed data.

    Every other node's data comes from upstream; something has to be the source, and this
    is it. In Phase 6 the planner takes over the job of deciding what those values are.
    """

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Return the node's ``value`` parameter.

        Raises:
            AgentError: If the node declares no ``value`` parameter.
        """
        if "value" not in ctx.params:
            raise AgentError(
                f"constant node {ctx.node_id!r} needs a 'value' parameter, "
                f"got params {sorted(ctx.params)}",
                # The definition will say exactly the same thing on the next attempt.
                retryable=False,
            )
        return ctx.params["value"]


@register("echo")
class EchoAgent:
    """Passes a single input straight through, or emits ``None`` if it has none."""

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Return the one input's value, or ``None`` when the node is a source."""
        if not ctx.inputs:
            return None
        (value,) = (ctx.inputs[name] for name in sorted(ctx.inputs))
        return value


class FailingAgent:
    """Always raises. Used to exercise the failure path without a flaky real agent.

    Raises a plain ``RuntimeError`` rather than an ``AgentError`` on purpose: that is what
    a bug in agent code looks like, and the default retry classification does not retry
    bugs. So this agent fails once and stays failed, whatever the attempt count says.

    Deliberately *not* registered: a workflow should never be able to name this by
    accident, so tests wire it explicitly into a registry of their own.
    """

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Raise, naming the node so the recorded error is traceable."""
        raise RuntimeError(f"node {ctx.node_id!r} was asked to fail")


class FlakyAgent:
    """Fails for the first few attempts, then succeeds.

    Deterministic rather than actually random — "flaky" in a test has to be reproducible.
    Keying on ``ctx.attempt`` rather than on a counter of its own is the point: it only
    ever succeeds if the executor really does thread a rising attempt number through to
    the agent, which is the same key Phase 5's idempotency will rest on.

    Not registered, for the same reason :class:`FailingAgent` is not.
    """

    def __init__(self, *, fail_until_attempt: int = 2) -> None:
        """Fail while ``ctx.attempt`` is below ``fail_until_attempt``."""
        self._fail_until_attempt = fail_until_attempt

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Return the attempt that finally worked.

        Raises:
            AgentError: On every attempt before ``fail_until_attempt``. Retryable, because
                this stands in for a provider having a moment rather than for a bug.
        """
        if ctx.attempt < self._fail_until_attempt:
            raise AgentError(
                f"node {ctx.node_id!r} failed on attempt {ctx.attempt}", retryable=True
            )
        return {"node_id": ctx.node_id, "attempt": ctx.attempt}


class SideEffectAgent:
    """Commits an effect to an outside world, exactly once, using the idempotency key.

    The engine can guarantee that a node is *re-executed* after a crash. It cannot
    guarantee that the node's effect on some other system did not already land — nothing
    on this side of a dead process can know that. What it can do is hand every attempt a
    stable name for the work, and this agent shows the pattern that name exists for:
    check, then commit under the key.

    That is DR-4 made concrete. Get this right and resume is just a reload; get it wrong
    and resume is a machine for double-charging people.

    Not registered: it needs a ledger injected, so a workflow file cannot name it.
    """

    def __init__(self, ledger: dict[str, NodeOutput]) -> None:
        """Commit into ``ledger``, standing in for whatever the real outside world is."""
        self.ledger = ledger
        self.commits = 0
        """How many times an effect was actually committed — the number a test asserts on."""

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Commit this node's effect if the key has not already been used."""
        if ctx.idempotency_key in self.ledger:
            # Already done, by an execution that may have died before saying so.
            return self.ledger[ctx.idempotency_key]

        # Annotated because JsonValue's members are invariant: an inferred dict[str, str]
        # is not a dict[str, JsonValue].
        effect: NodeOutput = {"node_id": ctx.node_id, "key": ctx.idempotency_key}
        self.ledger[ctx.idempotency_key] = effect
        self.commits += 1
        return effect


class HangingAgent:
    """Never returns, and records that its cancellation was delivered.

    A timeout is only *clean* if the coroutine actually receives its ``CancelledError`` at
    an await point and gets to run its cleanup. Counting that here is what lets a test
    assert cleanliness rather than merely observing that the executor moved on.

    Not registered: a workflow that could name this by accident would hang.
    """

    def __init__(self) -> None:
        """Start with nothing observed."""
        self.started = 0
        self.cancelled = 0
        self.entered = asyncio.Event()
        """Set once the agent is genuinely suspended, so a caller can wait rather than
        poll for the moment it becomes safe to cancel."""

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Block until cancelled.

        Raises:
            CancelledError: Always, once something cancels it. Re-raised rather than
                swallowed, which is what "cleanly cancelled" means for a coroutine.
        """
        self.started += 1
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return None  # pragma: no cover — the wait above never returns on its own
