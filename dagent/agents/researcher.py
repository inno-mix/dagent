"""An agent that investigates one topic with a single model call."""

from __future__ import annotations

from dagent.errors import AgentError
from dagent.models.model_call import ModelRequest
from dagent.models.state import NodeOutput
from dagent.runtime.agent import AgentContext
from dagent.runtime.registry import register

__all__ = ["ResearcherAgent"]

SYSTEM = (
    "You are a careful research assistant. Given a topic, state the most important "
    "things a well-informed reader should know. Be concrete and specific. Say plainly "
    "when something is uncertain or contested rather than smoothing it over."
)


@register("researcher")
class ResearcherAgent:
    """Turns a topic into findings.

    Reads whichever input the node declares — the agent does not care what its upstream
    node is called, only that exactly one input arrives. That is what lets the same agent
    sit under a static edge today and under a planner-generated one in Phase 6.
    """

    def __init__(self, *, max_output_tokens: int = 700, temperature: float = 0.3) -> None:
        """Configure the call this agent makes."""
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Research the incoming topic and return the findings.

        Raises:
            AgentError: If the node was not given exactly one input to research.
        """
        topic = _single_input(ctx, "researcher")
        response = await ctx.model.complete(
            ModelRequest(
                prompt=f"Topic: {topic}\n\nWhat should a well-informed reader know?",
                system=SYSTEM,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
            )
        )
        return {
            "topic": topic,
            "findings": response.text,
            "provider": response.provider,
            "model": response.model,
        }


def _single_input(ctx: AgentContext, agent_name: str) -> str:
    """Return the node's one input as text, or explain why that was not possible."""
    if len(ctx.inputs) != 1:
        raise AgentError(
            f"{agent_name} node {ctx.node_id!r} expects exactly one input, "
            f"got {sorted(ctx.inputs)}",
            # A wiring mistake in the definition, identical on every attempt.
            retryable=False,
        )
    (value,) = ctx.inputs.values()
    return value if isinstance(value, str) else str(value)
