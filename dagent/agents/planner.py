"""A planner: decomposes a question, then grows the graph to answer it (FR-7).

This is the agent SPEC §6's showcase workflow is built around. It asks a model to break a
question into subtopics, then emits one ``researcher`` node per subtopic plus a single
``synthesizer`` that fans them back in — so the *shape* of the run is decided at run time,
by the model, rather than written down in advance.

Nothing here knows how expansion works. It builds nodes and calls ``ctx.expand``; the
executor validates the augmented graph, merges it, persists it, and the ordinary ready-set
loop picks the new nodes up. That separation is the point: a planner is just an agent that
happens to return a graph-shaped opinion.
"""

from __future__ import annotations

from dagent.errors import AgentError
from dagent.graph.builder import build_node
from dagent.models.model_call import ModelRequest
from dagent.models.state import NodeOutput
from dagent.models.workflow import Node
from dagent.runtime.agent import AgentContext
from dagent.runtime.registry import register

__all__ = ["PlannerAgent"]

SYSTEM = (
    "You break a research question into independent subtopics. Reply with one subtopic "
    "per line and nothing else: no numbering, no bullets, no preamble. Each line must "
    "stand on its own as something a researcher could investigate without reading the "
    "others, because they will be researched in parallel."
)

DEFAULT_SUBTOPICS = 3
MAX_SUBTOPICS = 12
"""A hard stop on how far one planner may fan out.

Not a policy knob: it is the parsing layer refusing to turn a rambling model reply into
fifty concurrent API calls. The run-level ceiling in ``RunPolicy.max_graph_nodes`` is the
real bound; this is the one that keeps a single bad completion cheap.
"""


@register("planner")
class PlannerAgent:
    """Turns one question into N researchers and a synthesizer, at run time.

    The node ids it generates are derived from its own — ``<planner>.research_0``,
    ``<planner>.synthesis`` — rather than from the subtopic text or a counter. That makes
    the expansion a deterministic function of the planner's node id and the number of
    subtopics, so a planner re-executed after a crash restates the same nodes and the
    second expansion is a no-op instead of a second fan-out (DR-4).
    """

    def __init__(self, *, max_output_tokens: int = 300, temperature: float = 0.4) -> None:
        """Configure the call this agent makes."""
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Decompose the question, emit the nodes that answer it, and report the plan.

        Raises:
            AgentError: If the node was not given exactly one question, if ``subtopics``
                is not a positive integer, or if the model returned nothing usable.
        """
        question = _question(ctx)
        wanted = _subtopic_count(ctx)

        response = await ctx.model.complete(
            ModelRequest(
                prompt=f"Question: {question}\n\nGive {wanted} subtopics, one per line.",
                system=SYSTEM,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
            )
        )
        subtopics = _subtopics(response.text, wanted, node_id=ctx.node_id)
        ctx.expand(*_nodes_for(ctx.node_id, subtopics))

        return {
            "question": question,
            "subtopics": list(subtopics),
            "provider": response.provider,
            "model": response.model,
        }


def _nodes_for(planner_id: str, subtopics: tuple[str, ...]) -> tuple[Node, ...]:
    """Build the fan-out and the fan-in that answers a question.

    The synthesizer is emitted here rather than written into the workflow file because it
    has to depend on nodes that did not exist when the file was written. Expansion is
    append-only — an existing node can never gain a dependency — so the node that fans in
    must arrive with the nodes it fans in *from*.
    """
    researchers = tuple(
        build_node(
            f"{planner_id}.research_{index}",
            "researcher",
            depends_on=[planner_id],
            params={"topic": subtopic},
        )
        for index, subtopic in enumerate(subtopics)
    )
    synthesis = build_node(
        f"{planner_id}.synthesis",
        "synthesizer",
        inputs={node.id: node.id for node in researchers},
    )
    return (*researchers, synthesis)


def _question(ctx: AgentContext) -> str:
    """Read the one question this planner was given."""
    if len(ctx.inputs) != 1:
        raise AgentError(
            f"planner node {ctx.node_id!r} expects exactly one input, got {sorted(ctx.inputs)}",
            retryable=False,
        )
    (value,) = ctx.inputs.values()
    return value if isinstance(value, str) else str(value)


def _subtopic_count(ctx: AgentContext) -> int:
    """How many subtopics this planner was asked for."""
    wanted = ctx.params.get("subtopics", DEFAULT_SUBTOPICS)
    if not isinstance(wanted, int) or isinstance(wanted, bool) or not 1 <= wanted <= MAX_SUBTOPICS:
        raise AgentError(
            f"planner node {ctx.node_id!r} needs a 'subtopics' parameter between 1 and "
            f"{MAX_SUBTOPICS}, got {wanted!r}",
            retryable=False,
        )
    return wanted


def _subtopics(text: str, wanted: int, *, node_id: str) -> tuple[str, ...]:
    """Pull the subtopics out of a completion.

    One per line, rather than JSON: a model asked for prose in a fixed shape is far more
    reliable than one asked for a parseable object, and a stray blank line or a leading
    dash costs nothing here where a malformed brace would cost the whole node.

    Raises:
        AgentError: If nothing usable came back. Retryable — an empty completion is the
            kind of thing that works on the next attempt.
    """
    lines = [line.strip().lstrip("-*0123456789. ").strip() for line in text.splitlines()]
    subtopics = tuple(dict.fromkeys(line for line in lines if line))[:wanted]
    if not subtopics:
        raise AgentError(f"planner node {node_id!r} got no usable subtopics back from the model")
    return subtopics
