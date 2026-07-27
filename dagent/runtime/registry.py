"""Name → agent factory. Adding an agent requires no change to the core.

A registry is an ordinary object you can create, populate, and throw away, and the
executor takes one by injection. The module-level ``register`` decorator writes to a
shared default for convenience, but nothing in the engine reaches for that default —
tests build their own registry and never touch global state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, TypeVar

from dagent.errors import AgentError
from dagent.runtime.agent import Agent

__all__ = ["AgentFactory", "AgentRegistry", "default_registry", "register"]

AgentFactory: TypeAlias = Callable[[], Agent]
"""Anything that produces an agent when called — most often the agent class itself."""

F = TypeVar("F", bound=AgentFactory)


class AgentRegistry:
    """A mapping from registered agent name to the factory that builds one.

    Factories rather than instances, so each node gets a fresh agent and one node's
    state cannot leak into another's.
    """

    def __init__(self) -> None:
        """Start empty."""
        self._factories: dict[str, AgentFactory] = {}

    def register(self, name: str) -> Callable[[F], F]:
        """Return a decorator that registers a factory under ``name``.

        Raises:
            AgentError: If the name is already taken. Silently replacing an agent would
                make the behaviour of a workflow depend on import order.
        """

        def decorate(factory: F) -> F:
            self.add(name, factory)
            return factory

        return decorate

    def add(self, name: str, factory: AgentFactory) -> None:
        """Register a factory under ``name``.

        Raises:
            AgentError: If the name is already taken.
        """
        if not name:
            raise AgentError("an agent name cannot be empty")
        if name in self._factories:
            raise AgentError(f"agent {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name: str) -> Agent:
        """Build a fresh agent.

        Raises:
            AgentError: If no agent is registered under that name.
        """
        try:
            factory = self._factories[name]
        except KeyError:
            raise AgentError(f"no agent registered as {name!r}") from None
        return factory()

    def names(self) -> frozenset[str]:
        """Every registered name — this is what gets injected into graph validation."""
        return frozenset(self._factories)

    def __contains__(self, name: str) -> bool:
        """Whether an agent is registered under this name."""
        return name in self._factories

    def __len__(self) -> int:
        """How many agents are registered."""
        return len(self._factories)


default_registry = AgentRegistry()
"""The registry the module-level :func:`register` decorator writes to."""


def register(name: str) -> Callable[[F], F]:
    """Register an agent on the default registry.

    Convenience for application code that wants one global set of agents. Library and
    test code should build its own :class:`AgentRegistry` instead.
    """
    return default_registry.register(name)
