"""Concrete agents, and the provider clients they talk through.

This is the only package in dagent permitted to know about a vendor (AGENTS.md rule 3),
a boundary ``tests/test_isolation.py`` enforces mechanically. Everything upstream sees
only the ``Agent`` and ``ModelClient`` protocols.

Importing this package registers its agents on ``dagent.runtime.default_registry`` under
the names a workflow file uses: ``planner``, ``researcher``, ``synthesizer``,
``constant``, ``fake``, ``echo``. The agents built to misbehave or to need injection —
``FailingAgent``, ``FlakyAgent``, ``HangingAgent``, ``SideEffectAgent`` — are exported
but deliberately unregistered, so no workflow can name one by accident.
"""

from dagent.agents.fake import (
    ConstantAgent,
    EchoAgent,
    FailingAgent,
    FakeAgent,
    FlakyAgent,
    HangingAgent,
    SideEffectAgent,
)
from dagent.agents.gemini import GeminiClient
from dagent.agents.planner import PlannerAgent
from dagent.agents.researcher import ResearcherAgent
from dagent.agents.synthesizer import SynthesizerAgent

__all__ = [
    "ConstantAgent",
    "EchoAgent",
    "FailingAgent",
    "FakeAgent",
    "FlakyAgent",
    "GeminiClient",
    "HangingAgent",
    "PlannerAgent",
    "ResearcherAgent",
    "SideEffectAgent",
    "SynthesizerAgent",
]
