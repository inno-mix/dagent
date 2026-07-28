"""Schemas for one model call: what was asked, what came back, and the record of both.

Provider-agnostic on purpose. Nothing here mentions Gemini, and nothing here knows how a
request is transported — that belongs to a client in ``dagent/agents``. Keeping the
request and response as plain frozen schemas is what lets the recording wrapper write
them straight into the run record, and what will let a replay client serve them back
without a network (FR-8).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelCallRecord", "ModelRequest", "ModelResponse"]


class ModelRequest(BaseModel):
    """One single-turn request to a language model.

    Single-turn rather than a message list because that is all any agent in this project
    needs so far, and a narrower schema is a smaller thing to keep stable across
    providers. Multi-turn widens this rather than replaces it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str = Field(min_length=1)
    system: str | None = None
    model: str | None = None
    """Which model to use, or ``None`` to accept the client's configured default."""
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, gt=0)


class ModelResponse(BaseModel):
    """What a model returned, plus the usage that Phase 4's budget will meter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    provider: str
    model: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        """Tokens charged for this call."""
        return self.input_tokens + self.output_tokens


class ModelCallRecord(BaseModel):
    """A single model call as persisted into the run record.

    ``(run_id, node_id, attempt, sequence)`` identifies a call uniquely and in order,
    which is exactly what a replay client needs to serve the right recorded response to
    the right request without a network (FR-8).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    node_id: str
    attempt: int = Field(default=0, ge=0)
    sequence: int = Field(default=0, ge=0)
    """Position of this call within its node attempt, counted from zero."""
    request: ModelRequest
    response: ModelResponse
    started_at: datetime | None = None
    finished_at: datetime | None = None
