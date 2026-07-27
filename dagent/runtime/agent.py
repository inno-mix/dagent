"""The agent contract: one async method, and the context it is handed.

An agent is the pluggable behaviour behind a node — an LLM call, a tool, or pure logic.
The core knows nothing about which. Everything non-deterministic an agent needs arrives
through ``AgentContext`` rather than being reached for directly, which is what makes a
run replayable and a test network-free (DR-5).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from dagent.models.state import NodeOutput
from dagent.runtime.clock import Clock

__all__ = ["Agent", "AgentContext"]


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Everything a node's agent is given when it runs.

    A plain frozen dataclass rather than a pydantic model: it carries live injected
    objects like the clock, which are behaviour rather than data and have no business
    being validated or serialized.

    Later phases widen this rather than reshape it — the model client lands in Phase 3
    and the budget handle in Phase 4, both as further injected seams.
    """

    run_id: str
    node_id: str
    attempt: int
    inputs: Mapping[str, NodeOutput]
    clock: Clock


class Agent(Protocol):
    """The single method every agent implements.

    A ``Protocol`` rather than a base class (AGENTS.md §5): an agent needs no import from
    dagent to satisfy it, so adding one requires no change to the core.
    """

    async def run(self, ctx: AgentContext) -> NodeOutput:
        """Do the node's work and return its serializable output."""
        ...
