"""The agent contract: one async method, and the context it is handed.

An agent is the pluggable behaviour behind a node — an LLM call, a tool, or pure logic.
The core knows nothing about which. Everything non-deterministic an agent needs arrives
through ``AgentContext`` rather than being reached for directly, which is what makes a
run replayable and a test network-free (DR-5).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from dagent.models.state import NodeOutput
from dagent.models.workflow import Node
from dagent.runtime.clock import Clock
from dagent.runtime.expansion import Expansion
from dagent.runtime.model import ModelClient, NullModelClient

__all__ = ["Agent", "AgentContext"]


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Everything a node's agent is given when it runs.

    A plain frozen dataclass rather than a pydantic model: it carries live injected
    objects like the clock, which are behaviour rather than data and have no business
    being validated or serialized.

    Later phases widen this rather than reshape it: Phase 4 added the model seam's
    metering, Phase 5 the idempotency key, Phase 6 the expansion handle.
    """

    run_id: str
    node_id: str
    attempt: int
    inputs: Mapping[str, NodeOutput]
    clock: Clock
    params: Mapping[str, NodeOutput] = field(default_factory=dict)
    """This node's static configuration, copied straight from its definition."""
    model: ModelClient = field(default_factory=NullModelClient)
    """The model seam.

    Always present, so an agent never branches on ``None``. A run that wires no model
    gets a client that refuses loudly, which turns "forgot to configure a provider" into
    a clear error rather than a crash inside agent code.
    """

    expansion: Expansion = field(default_factory=Expansion)
    """Where a request to grow the graph is collected. Use :meth:`expand` rather than this."""

    def expand(self, *nodes: Node) -> None:
        """Ask the executor to add these nodes to the running graph (FR-7).

        Nothing happens immediately. The request is collected, and applied only once this
        node succeeds — so an attempt that expands and *then* fails leaves no trace, and
        the retry that follows starts from a graph nobody has already grown.

        The executor validates the augmented graph before merging it. An expansion that
        would introduce a cycle, name an unregistered agent, redefine an existing node, or
        run past the depth limit fails *this node* and leaves the rest of the run alone.

        Build the nodes with :func:`~dagent.graph.builder.build_node`, which fills in the
        ``depends_on`` edges implied by their inputs — the same thing ``WorkflowBuilder``
        does, so a generated node is held to exactly the standard a written one is.
        """
        self.expansion.add(*nodes)

    @property
    def idempotency_key(self) -> str:
        """A stable name for *this* unit of work, for deduplicating side effects.

        ``(run_id, node_id, attempt)``, rendered as a string so it drops straight into
        whatever the outside world uses for the job — a payment provider's idempotency
        header, a unique index, an object key.

        The guarantee that makes it useful: an interrupted node re-dispatched by
        :meth:`~dagent.runtime.executor.Executor.resume` re-runs under the *same* attempt
        number, and therefore the same key. A crash cannot tell you whether the side
        effect landed before the process died, so the engine assumes it might have and
        hands back a key the outside world can recognise. A retry after a *definite*
        failure is a different matter and gets a new attempt, because that work provably
        did not complete.

        This is DR-4 in one line: re-execution is made safe first, which is what leaves
        resume with nothing to do but reload.
        """
        return f"{self.run_id}:{self.node_id}:{self.attempt}"


class Agent(Protocol):
    """The single method every agent implements.

    A ``Protocol`` rather than a base class (AGENTS.md §5): an agent needs no import from
    dagent to satisfy it, so adding one requires no change to the core.
    """

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Do the node's work and return its serializable output."""
        ...
