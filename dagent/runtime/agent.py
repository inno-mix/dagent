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
from dagent.runtime.clock import Clock
from dagent.runtime.model import ModelClient, NullModelClient

__all__ = ["Agent", "AgentContext"]


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Everything a node's agent is given when it runs.

    A plain frozen dataclass rather than a pydantic model: it carries live injected
    objects like the clock, which are behaviour rather than data and have no business
    being validated or serialized.

    Later phases widen this rather than reshape it — the budget handle lands in Phase 4
    as a further injected seam.
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


class Agent(Protocol):
    """The single method every agent implements.

    A ``Protocol`` rather than a base class (AGENTS.md §5): an agent needs no import from
    dagent to satisfy it, so adding one requires no change to the core.
    """

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Do the node's work and return its serializable output."""
        ...
