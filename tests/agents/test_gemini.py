"""The Gemini client, exercised through httpx's MockTransport.

The real request-building and response-parsing code runs; only the socket is replaced.
That is the difference between testing the client and testing a mock of it — and it is
how this stays a test that never touches the network (AGENTS.md §6).
"""

import httpx
import pytest

from dagent.agents.gemini import API_KEY_ENV, DEFAULT_MODEL, MODEL_ENV, GeminiClient
from dagent.errors import AgentError, DagentError
from dagent.models.model_call import ModelRequest
from dagent.runtime.model import ModelClient

OK_BODY = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": "the answer"}]}, "finishReason": "STOP"}
    ],
    "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 4},
    "modelVersion": "gemini-2.5-flash-001",
}


def client_with(handler) -> GeminiClient:  # type: ignore[no-untyped-def]
    return GeminiClient(
        api_key="test-key", http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def always(body: object, status: int = 200):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return handler


def test_the_client_satisfies_the_model_client_protocol() -> None:
    assert isinstance(GeminiClient(api_key="k"), ModelClient)


def test_the_provider_name_is_what_per_provider_caps_will_key_on() -> None:
    assert GeminiClient(api_key="k").provider == "gemini"


def test_an_empty_api_key_is_refused_up_front() -> None:
    with pytest.raises(AgentError, match=API_KEY_ENV):
        GeminiClient(api_key="")


def test_from_env_reads_the_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "from-env")

    assert GeminiClient.from_env().provider == "gemini"


def test_from_env_refuses_when_the_key_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keys come from the environment and nowhere else (AGENTS.md rule 7).
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    with pytest.raises(AgentError, match=API_KEY_ENV):
        GeminiClient.from_env()


def test_from_env_honours_a_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "k")
    monkeypatch.setenv(MODEL_ENV, "gemini-experimental")
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json=OK_BODY)

    client = GeminiClient.from_env(http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    import asyncio

    asyncio.run(client.complete(ModelRequest(prompt="hi")))
    assert "gemini-experimental:generateContent" in captured["url"]


@pytest.mark.asyncio
async def test_a_successful_call_returns_text_and_usage() -> None:
    response = await client_with(always(OK_BODY)).complete(ModelRequest(prompt="what?"))

    assert response.text == "the answer"
    assert response.provider == "gemini"
    assert response.model == "gemini-2.5-flash-001"
    assert (response.input_tokens, response.output_tokens) == (11, 4)
    assert response.total_tokens == 15


@pytest.mark.asyncio
async def test_the_request_carries_the_prompt_and_generation_config() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    await client_with(handler).complete(
        ModelRequest(prompt="the prompt", temperature=0.7, max_output_tokens=64)
    )

    body = seen["body"]
    assert isinstance(body, dict)
    assert body["contents"] == [{"role": "user", "parts": [{"text": "the prompt"}]}]
    assert body["generationConfig"]["temperature"] == 0.7
    assert body["generationConfig"]["maxOutputTokens"] == 64
    assert "systemInstruction" not in body
    assert f"{DEFAULT_MODEL}:generateContent" in str(seen["url"])


@pytest.mark.asyncio
async def test_thinking_is_disabled_by_default() -> None:
    # Gemini 2.5 charges internal reasoning against maxOutputTokens, so leaving thinking
    # on returns prose truncated mid-sentence. Found by actually running it.
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    await client_with(handler).complete(ModelRequest(prompt="hi"))

    body = seen["body"]
    assert isinstance(body, dict)
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


@pytest.mark.asyncio
async def test_a_thinking_budget_can_be_raised() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    client = GeminiClient(
        api_key="k",
        thinking_budget=512,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await client.complete(ModelRequest(prompt="hi"))

    body = seen["body"]
    assert isinstance(body, dict)
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 512}


@pytest.mark.asyncio
async def test_the_thinking_field_is_omitted_entirely_when_set_to_none() -> None:
    # Required for models that cannot disable thinking — sending the field would error.
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    client = GeminiClient(
        api_key="k",
        thinking_budget=None,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await client.complete(ModelRequest(prompt="hi"))

    body = seen["body"]
    assert isinstance(body, dict)
    assert "thinkingConfig" not in body["generationConfig"]


@pytest.mark.asyncio
async def test_the_api_key_travels_as_a_header_not_in_the_url() -> None:
    # A key in a query string ends up in access logs and error messages.
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json=OK_BODY)

    await client_with(handler).complete(ModelRequest(prompt="hi"))

    assert seen["key"] == "test-key"
    assert "test-key" not in str(seen["url"])


@pytest.mark.asyncio
async def test_a_system_instruction_is_sent_when_given() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    await client_with(handler).complete(ModelRequest(prompt="hi", system="be terse"))

    body = seen["body"]
    assert isinstance(body, dict)
    assert body["systemInstruction"] == {"parts": [{"text": "be terse"}]}


@pytest.mark.asyncio
async def test_a_request_level_model_overrides_the_client_default() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=OK_BODY)

    await client_with(handler).complete(ModelRequest(prompt="hi", model="gemini-2.5-pro"))

    assert "gemini-2.5-pro:generateContent" in str(seen["url"])


@pytest.mark.asyncio
async def test_an_http_error_status_becomes_an_agent_error() -> None:
    with pytest.raises(AgentError, match="HTTP 429"):
        await client_with(always({"error": "rate limited"}, status=429)).complete(
            ModelRequest(prompt="hi")
        )


@pytest.mark.asyncio
async def test_a_transport_failure_becomes_an_agent_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with pytest.raises(AgentError, match="gemini request failed"):
        await client_with(handler).complete(ModelRequest(prompt="hi"))


@pytest.mark.asyncio
async def test_a_blocked_prompt_becomes_an_agent_error_naming_the_reason() -> None:
    body = {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}

    with pytest.raises(AgentError, match="SAFETY"):
        await client_with(always(body)).complete(ModelRequest(prompt="hi"))


@pytest.mark.asyncio
async def test_an_empty_completion_becomes_an_agent_error_naming_the_finish_reason() -> None:
    body = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}

    with pytest.raises(AgentError, match="MAX_TOKENS"):
        await client_with(always(body)).complete(ModelRequest(prompt="hi"))


@pytest.mark.asyncio
async def test_multi_part_responses_are_joined() -> None:
    body = {
        "candidates": [{"content": {"parts": [{"text": "one "}, {"text": "two"}]}}],
        "usageMetadata": {},
    }

    response = await client_with(always(body)).complete(ModelRequest(prompt="hi"))

    assert response.text == "one two"
    assert response.input_tokens == 0


@pytest.mark.asyncio
async def test_every_failure_is_catchable_as_a_dagent_error() -> None:
    with pytest.raises(DagentError):
        await client_with(always({}, status=500)).complete(ModelRequest(prompt="hi"))


@pytest.mark.asyncio
async def test_a_borrowed_http_client_is_not_closed_by_this_class() -> None:
    # Whoever opened the connection pool owns it; closing someone else's is a bug.
    borrowed = httpx.AsyncClient(transport=httpx.MockTransport(always(OK_BODY)))
    client = GeminiClient(api_key="k", http=borrowed)

    await client.aclose()

    assert not borrowed.is_closed
    await borrowed.aclose()


@pytest.mark.asyncio
async def test_an_owned_http_client_is_closed_on_context_exit() -> None:
    client = GeminiClient(api_key="k")

    async with client:
        pass

    with pytest.raises(RuntimeError):
        await client.complete(ModelRequest(prompt="hi"))
