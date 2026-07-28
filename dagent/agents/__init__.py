"""Concrete agents, and the provider clients they talk through.

This is the only package in dagent permitted to know about a vendor (AGENTS.md rule 3),
a boundary ``tests/test_isolation.py`` enforces mechanically. Everything upstream sees
only the ``Agent`` and ``ModelClient`` protocols.

Importing this package registers its agents on ``dagent.runtime.default_registry`` under
the names a workflow file uses: ``researcher``, ``synthesizer``, ``fake``, ``echo``.
"""

from dagent.agents.fake import EchoAgent, FailingAgent, FakeAgent
from dagent.agents.gemini import GeminiClient
from dagent.agents.researcher import ResearcherAgent
from dagent.agents.synthesizer import SynthesizerAgent

__all__ = [
    "EchoAgent",
    "FailingAgent",
    "FakeAgent",
    "GeminiClient",
    "ResearcherAgent",
    "SynthesizerAgent",
]
