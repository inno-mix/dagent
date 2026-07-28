"""The model seam: the one interface agents call, and the doubles that stand in for it.

This module is provider-agnostic and imports no HTTP client and no model SDK. A concrete
provider lives in ``dagent/agents`` — the only package allowed provider-specific code —
so the core can schedule model-backed work without ever depending on a vendor.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from dagent.errors import AgentError
from dagent.models.model_call import ModelRequest, ModelResponse

__all__ = ["ModelClient", "NullModelClient", "StubModelClient"]


@runtime_checkable
class ModelClient(Protocol):
    """The single method an agent uses to talk to a language model.

    Deliberately one method. Everything that varies between providers — auth, endpoint
    shape, retry-worthy status codes — is the implementation's problem, not the agent's.
    """

    @property
    def provider(self) -> str:
        """Short provider name, used for per-provider concurrency caps and metrics."""
        ...

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send one request and return what the model produced."""
        ...


class NullModelClient:
    """The default client: refuses every call, loudly.

    A run that wires no model still gives every agent a client, so an agent never has to
    branch on ``None``. An agent that then tries to use a model it was never given gets a
    clear error naming the omission instead of an ``AttributeError`` three frames deep.
    """

    @property
    def provider(self) -> str:
        """There is no provider."""
        return "null"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Always raise.

        Raises:
            AgentError: Always. This run was not configured with a model client.
        """
        raise AgentError(
            "this agent asked for a model call but the run has no model client "
            "configured; pass one to Executor(model=...)"
        )


class StubModelClient:
    """A scripted client for tests and examples. Never touches the network.

    Give it replies and it hands them out in order, recording every request it saw. This
    is what "the same workflow runs in tests with zero network" rests on — the agent code
    under test is the real code, only the transport is replaced.
    """

    def __init__(
        self,
        replies: Sequence[str] | Callable[[ModelRequest], str] = ("stub response",),
        *,
        provider: str = "stub",
        model: str = "stub-model",
    ) -> None:
        """Script the replies, either as a fixed sequence or as a function of the request."""
        self._replies = replies
        self._provider = provider
        self._model = model
        self.requests: list[ModelRequest] = []

    @property
    def provider(self) -> str:
        """The provider name this stub reports as."""
        return self._provider

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the next scripted reply.

        Raises:
            AgentError: If a fixed script runs out of replies. Silently repeating the
                last one would let a test pass while the agent made more calls than it
                was supposed to.
        """
        self.requests.append(request)

        if callable(self._replies):
            text = self._replies(request)
        else:
            index = len(self.requests) - 1
            if index >= len(self._replies):
                raise AgentError(
                    f"StubModelClient was scripted with {len(self._replies)} replies "
                    f"but received {len(self.requests)} requests"
                )
            text = self._replies[index]

        return ModelResponse(
            text=text,
            provider=self._provider,
            model=request.model or self._model,
            input_tokens=len(request.prompt.split()),
            output_tokens=len(text.split()),
        )
