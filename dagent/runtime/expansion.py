"""Growing the graph while it runs — FR-7, and the second-hardest thing in the project.

Two pieces. :class:`Expansion` is what an agent fills in: a request to add nodes, handed
to it on its context and collected while it runs. :class:`RunGraph` is what the executor
holds: the live definition, and the one place a request is checked and merged.

The invariant everything else rests on is that :meth:`RunGraph.apply` contains **no
``await``**. On a single event loop that makes validate-and-merge atomic with respect to
every other node: two planners finishing in the same batch are applied one after the
other, and the second is validated against a graph that already includes the first. No
lock, no queue, no window in which the graph is half-expanded — the absence of a
suspension point *is* the serialisation ARCHITECTURE §4 promises.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field

from dagent.errors import ValidationError
from dagent.graph.expansion import expand_workflow
from dagent.graph.topo import nodes_by_id
from dagent.models.workflow import Node, Workflow

__all__ = ["Expansion", "RunGraph"]


@dataclass(slots=True)
class Expansion:
    """A node's request to add nodes to the running graph.

    Mutable, and one per node *attempt* — a request from an attempt that then failed is
    thrown away with the attempt, so a retry starts from a clean slate rather than
    expanding the graph twice.
    """

    nodes: list[Node] = field(default_factory=list)

    def add(self, *nodes: Node) -> None:
        """Ask for these nodes to be added once this node succeeds."""
        self.nodes.extend(nodes)

    def __bool__(self) -> bool:
        """Whether anything was requested."""
        return bool(self.nodes)


class RunGraph:
    """The definition one run is executing, and the only thing allowed to change it.

    A ``Workflow`` is frozen (FR-1) and stays frozen: expansion replaces the whole object
    rather than editing one, so anything holding a reference to the old graph keeps seeing
    a consistent — if stale — picture. This class holds the current one.
    """

    def __init__(
        self,
        workflow: Workflow,
        *,
        known_agents: Collection[str] | None = None,
        max_depth: int = 1,
        max_nodes: int = 1000,
    ) -> None:
        """Hold ``workflow`` and the limits an expansion of it must respect.

        Args:
            workflow: The definition as submitted, or as reloaded on resume.
            known_agents: Registered agent names, forwarded to the validator.
            max_depth: How many generations of expansion are allowed. Every node present
                at construction is generation zero.
            max_nodes: A ceiling on the whole graph, expanded or not.
        """
        self._workflow = workflow
        self._known_agents = known_agents
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._depth: dict[str, int] = dict.fromkeys(nodes_by_id(workflow), 0)

    @property
    def workflow(self) -> Workflow:
        """The definition as it stands now."""
        return self._workflow

    def apply(self, source: str, expansion: Expansion) -> tuple[Node, ...]:
        """Validate an expansion and merge it, or reject it whole.

        Deliberately synchronous. See the module docstring: no ``await`` here is what
        makes this atomic on one event loop.

        Args:
            source: The node that asked, which is what bounds the expansion's depth.
            expansion: What it asked for.

        Returns:
            The nodes actually added — empty when the request only restated nodes the
            graph already has.

        Raises:
            ValidationError: If the request would exceed a bound, redefine an existing
                node, or leave the graph invalid. Nothing is merged in that case, so a
                rejected expansion leaves the run exactly as it was. The caller turns this
                into an ordinary node failure, which is why a bad planner cannot deadlock
                a run — it just fails, and the graph carries on without it.
        """
        if not expansion:
            return ()

        depth = self._depth.get(source, 0) + 1
        if depth > self._max_depth:
            raise ValidationError(
                f"node {source!r} tried to expand at depth {depth}, above the limit of "
                f"{self._max_depth}; raise RunPolicy.max_expansion_depth to allow it"
            )

        before = nodes_by_id(self._workflow)
        total = len(before) + len(expansion.nodes)
        if total > self._max_nodes:
            raise ValidationError(
                f"node {source!r} would grow the graph to as many as {total} nodes, above "
                f"the limit of {self._max_nodes}"
            )

        # The merge rule itself is pure and lives in `graph`; what this class adds is the
        # run-level context that rule must not depend on — who asked, how deep they are,
        # and how big the graph is allowed to get.
        augmented = expand_workflow(
            self._workflow, expansion.nodes, known_agents=self._known_agents
        )
        if augmented is self._workflow:
            return ()

        self._workflow = augmented
        added = tuple(node for node in augmented.nodes if node.id not in before)
        for node in added:
            self._depth[node.id] = depth
        return added
