"""Agents that do no I/O — for examples, and for exercising the engine without a model.

Shipped rather than confined to the test suite because anyone building a workflow needs
to run its shape before its prompts exist, and because every test in this repository is
required to be network-free (AGENTS.md §6).
"""

from __future__ import annotations

from dagent.models.state import NodeOutput
from dagent.runtime.agent import AgentContext

__all__ = ["EchoAgent", "FailingAgent", "FakeAgent"]


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


class EchoAgent:
    """Passes a single input straight through, or emits ``None`` if it has none."""

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Return the one input's value, or ``None`` when the node is a source."""
        if not ctx.inputs:
            return None
        (value,) = (ctx.inputs[name] for name in sorted(ctx.inputs))
        return value


class FailingAgent:
    """Always raises. Used to exercise the failure path without a flaky real agent."""

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Raise, naming the node so the recorded error is traceable."""
        raise RuntimeError(f"node {ctx.node_id!r} was asked to fail")
