"""The agent contract, the agent registry, the time seam, and the async executor.

``executor.py`` is the heart of the project and imports no model SDK: it schedules
opaque agents and knows nothing about what any of them do.
"""

from dagent.runtime.agent import Agent, AgentContext
from dagent.runtime.clock import Clock, ManualClock, SystemClock
from dagent.runtime.executor import Executor
from dagent.runtime.metering import BudgetedModelClient, Pricer
from dagent.runtime.model import ModelClient, NullModelClient, StubModelClient
from dagent.runtime.recording import RecordingModelClient
from dagent.runtime.registry import (
    AgentFactory,
    AgentRegistry,
    default_registry,
    register,
)

__all__ = [
    "Agent",
    "AgentContext",
    "AgentFactory",
    "AgentRegistry",
    "BudgetedModelClient",
    "Clock",
    "Executor",
    "ManualClock",
    "ModelClient",
    "NullModelClient",
    "Pricer",
    "RecordingModelClient",
    "StubModelClient",
    "SystemClock",
    "default_registry",
    "register",
]
