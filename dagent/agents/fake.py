"""Agents that do no I/O — for examples, and for exercising the engine without a model.

Shipped rather than confined to the test suite because anyone building a workflow needs
to run its shape before its prompts exist, and because every test in this repository is
required to be network-free (AGENTS.md §6).
"""

from __future__ import annotations

from dagent.errors import AgentError
from dagent.models.state import NodeOutput
from dagent.runtime.agent import AgentContext
from dagent.runtime.registry import register

__all__ = ["ConstantAgent", "EchoAgent", "FailingAgent", "FakeAgent"]


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
                f"got params {sorted(ctx.params)}"
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

    Deliberately *not* registered: a workflow should never be able to name this by
    accident, so tests wire it explicitly into a registry of their own.
    """

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Raise, naming the node so the recorded error is traceable."""
        raise RuntimeError(f"node {ctx.node_id!r} was asked to fail")
