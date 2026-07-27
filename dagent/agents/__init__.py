"""Concrete agents: planner, researcher, synthesizer, critic — plus the no-I/O fakes.

This is the only package in dagent permitted to import a model SDK (AGENTS.md rule 3), a
boundary ``tests/test_isolation.py`` enforces mechanically. The real LLM agents arrive in
Phase 3; nothing here touches the network yet.
"""

from dagent.agents.fake import EchoAgent, FailingAgent, FakeAgent

__all__ = ["EchoAgent", "FailingAgent", "FakeAgent"]
