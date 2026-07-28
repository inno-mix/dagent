"""A ``ModelClient`` for Google's Gemini API, over plain HTTP.

Provider-specific code lives here because ``dagent/agents`` is the only package allowed
to know about a vendor (AGENTS.md rule 3). Everything upstream of this file talks to the
:class:`~dagent.runtime.model.ModelClient` protocol and has no idea which provider — or
whether there is one at all.

Deliberately no vendor SDK: the Gemini REST surface this needs is one POST, and going
through ``httpx`` directly keeps the dependency list honest and the request/response
shape visible in the code rather than behind a client library.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Any, Self

import httpx

from dagent.errors import AgentError
from dagent.models.model_call import ModelRequest, ModelResponse

__all__ = ["GeminiClient"]

API_KEY_ENV = "GEMINI_API_KEY"
MODEL_ENV = "DAGENT_GEMINI_MODEL"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT = 60.0

DEFAULT_THINKING_BUDGET = 0
"""Reasoning tokens allowed per call, defaulting to none.

Gemini 2.5 models charge internal reasoning against ``maxOutputTokens``, so a request for
700 tokens of prose can return two truncated sentences with the rest spent thinking. The
seam's ``max_output_tokens`` means "how much text I want back", and zeroing the thinking
budget is what makes Gemini honour that. Models that cannot disable thinking need
``thinking_budget=None`` so the field is omitted.
"""


class GeminiClient:
    """Talks to Gemini's ``generateContent`` endpoint over a shared ``httpx.AsyncClient``.

    One client per run, not per call (AGENTS.md §4): connection reuse is most of the
    latency budget when a workflow fans out to a dozen concurrent nodes.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        http: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        thinking_budget: int | None = DEFAULT_THINKING_BUDGET,
    ) -> None:
        """Build a client.

        Args:
            api_key: The Gemini API key. Never logged, never included in an error.
            model: Default model, used when a request does not name one.
            base_url: API root, overridable for testing against a stub server.
            http: An existing client to borrow. Supply one and this class will not close
                it — whoever opened it owns it.
            timeout: Per-request timeout in seconds.
            thinking_budget: Tokens the model may spend on internal reasoning, which on
                Gemini 2.5 come out of ``max_output_tokens``. Defaults to ``0`` — see
                :data:`DEFAULT_THINKING_BUDGET` for why. Pass ``None`` to omit the field
                entirely, which is required for models that cannot disable thinking.

        Raises:
            AgentError: If the API key is empty.
        """
        if not api_key:
            raise AgentError(f"a Gemini API key is required; set {API_KEY_ENV}")

        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._thinking_budget = thinking_budget
        self._owns_http = http is None
        self._http = http if http is not None else httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(cls, *, http: httpx.AsyncClient | None = None) -> Self:
        """Build a client from the environment.

        Keys come from the environment and nowhere else (AGENTS.md rule 7) — never from a
        config file, an argument default, or a fixture.

        Raises:
            AgentError: If ``GEMINI_API_KEY`` is unset or empty.
        """
        api_key = os.environ.get(API_KEY_ENV, "")
        if not api_key:
            raise AgentError(
                f"{API_KEY_ENV} is not set; copy .env.example to .env and fill it in, "
                "then run under `uv run --env-file .env`"
            )
        return cls(
            api_key=api_key,
            model=os.environ.get(MODEL_ENV) or DEFAULT_MODEL,
            http=http,
        )

    @property
    def provider(self) -> str:
        """The name Phase 4's per-provider concurrency cap will key on."""
        return "gemini"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send one request to Gemini and return the text it produced.

        Raises:
            AgentError: If the request fails, or the response carries no usable text —
                which is what an empty candidate list means when a prompt is filtered.
        """
        model = request.model or self._model
        generation: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_output_tokens,
        }
        if self._thinking_budget is not None:
            generation["thinkingConfig"] = {"thinkingBudget": self._thinking_budget}

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": generation,
        }
        if request.system is not None:
            payload["systemInstruction"] = {"parts": [{"text": request.system}]}

        try:
            response = await self._http.post(
                f"{self._base_url}/models/{model}:generateContent",
                json=payload,
                headers={"x-goog-api-key": self._api_key},
            )
        except httpx.HTTPError as exc:
            raise AgentError(f"gemini request failed: {type(exc).__name__}: {exc}") from exc

        if response.status_code != httpx.codes.OK:
            # response.text can echo the request but never the key: it is a header.
            raise AgentError(
                f"gemini returned HTTP {response.status_code}: {_truncate(response.text)}"
            )

        return _parse_response(response.json(), fallback_model=model)

    async def aclose(self) -> None:
        """Close the underlying HTTP client, if this instance opened it."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> Self:
        """Enter a context that closes the HTTP client on the way out."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the HTTP client."""
        await self.aclose()


def _parse_response(body: Any, *, fallback_model: str) -> ModelResponse:
    """Pull the text and token usage out of a ``generateContent`` response."""
    candidates = body.get("candidates") or []
    if not candidates:
        reason = body.get("promptFeedback", {}).get("blockReason", "no candidates returned")
        raise AgentError(f"gemini produced no output: {reason}")

    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    if not text:
        finish = candidates[0].get("finishReason", "unknown")
        raise AgentError(f"gemini produced empty text (finishReason={finish})")

    usage = body.get("usageMetadata") or {}
    return ModelResponse(
        text=text,
        provider="gemini",
        model=body.get("modelVersion") or fallback_model,
        input_tokens=usage.get("promptTokenCount", 0),
        output_tokens=usage.get("candidatesTokenCount", 0),
    )


def _truncate(text: str, limit: int = 500) -> str:
    """Keep an error message readable when a provider returns a wall of HTML."""
    return text if len(text) <= limit else f"{text[:limit]}..."
