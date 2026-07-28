"""An agent that merges several upstream results into one answer — the fan-in worker."""

from __future__ import annotations

import json

from dagent.errors import AgentError
from dagent.models.model_call import ModelRequest
from dagent.models.state import NodeOutput
from dagent.runtime.agent import AgentContext
from dagent.runtime.registry import register

__all__ = ["SynthesizerAgent"]

SYSTEM = (
    "You are a synthesist. You are given several independent research notes. Merge them "
    "into one coherent account: keep what they agree on, name where they disagree, and "
    "do not invent anything that is not in the notes."
)


@register("synthesizer")
class SynthesizerAgent:
    """Merges every input it is given into a single narrative.

    Takes whatever arrives rather than a fixed set of names, so the same agent works
    behind two static branches today and behind N planner-generated ones in Phase 6.
    Inputs are sorted before prompting, so the same set of upstream outputs always
    produces the same prompt — the prerequisite for a replayable run.
    """

    def __init__(self, *, max_output_tokens: int = 900, temperature: float = 0.2) -> None:
        """Configure the call this agent makes."""
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Merge the incoming notes into one answer.

        Raises:
            AgentError: If the node was given nothing to synthesize.
        """
        if not ctx.inputs:
            raise AgentError(
                f"synthesizer node {ctx.node_id!r} has no inputs to merge; "
                "it needs at least one upstream node",
                # A wiring mistake in the definition, identical on every attempt.
                retryable=False,
            )

        sections = [
            f"--- note from {name} ---\n{_as_text(value)}"
            for name, value in sorted(ctx.inputs.items())
        ]
        response = await ctx.model.complete(
            ModelRequest(
                prompt="\n\n".join(
                    [*sections, "\n--- end of notes ---\n\nWrite the merged account."]
                ),
                system=SYSTEM,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
            )
        )
        # Annotated because JsonValue's list member is invariant: an inferred list[str]
        # is not a list[JsonValue], and the annotation is what makes the names fit.
        sources: list[NodeOutput] = [*sorted(ctx.inputs)]
        return {
            "summary": response.text,
            "sources": sources,
            "provider": response.provider,
            "model": response.model,
        }


def _as_text(value: NodeOutput) -> str:
    """Render an upstream output for the prompt.

    A researcher's dict is unwrapped to its findings so the prompt reads as prose rather
    than as JSON; anything else is serialized deterministically.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("findings"), str):
        return str(value["findings"])
    return json.dumps(value, sort_keys=True)
