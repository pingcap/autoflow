import asyncio
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import dspy
from dspy.lm15 import Message, Request, TextDelta
from dspy.utils.exceptions import LMUnexpectedError

from app.core.config import settings
from app.rag.llms.dspy import get_dspy_lm_by_llama_llm
from app.rag.llms.provider import LLMProvider
from app.rag.llms.resolver import resolve_llm
from app.security.generation import GenerationScope, current_generation
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError


@pytest.fixture(params=[LLMProvider.OPENAI, LLMProvider.AZURE_OPENAI])
def provider(monkeypatch, request):
    state = SimpleNamespace(
        calls=[], bodies=[], auth=[], mode="normal", disconnected=Event()
    )
    state.azure = request.param == LLMProvider.AZURE_OPENAI

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            state.bodies.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            authorization = self.headers.get("Authorization")
            api_key = self.headers.get("api-key")
            state.auth.append((authorization, api_key))
            state.calls.append((self.path, api_key if state.azure else authorization))
            content = "fixture answer"
            if state.mode == "structured":
                content = '{"answer": "fixture answer"}'
            if state.mode == "oversized":
                content = "x" * (4 * 1024 * 1024 + 1)
            body = json.dumps(
                {
                    "id": "fixture",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode()
            streaming = state.bodies[-1].get("stream", False)
            if streaming:
                chunk = {
                    "id": "fixture",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": content}}
                    ],
                }
                end = {
                    "id": "fixture",
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                }
                body = (
                    "data: "
                    + json.dumps(chunk)
                    + "\n\ndata: "
                    + json.dumps(end)
                    + "\n\ndata: [DONE]\n\n"
                ).encode()
            self.send_response(302 if state.mode == "redirect" else 200)
            self.send_header(
                "Content-Type", "text/event-stream" if streaming else "application/json"
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            if state.mode == "compressed":
                self.send_header("Content-Encoding", "gzip")
            if state.mode == "redirect":
                self.send_header("Location", state.origin + "/should-not-follow")
            self.end_headers()
            try:
                if state.mode == "slow":
                    # Bytes keep arriving within the socket timeout; the shared
                    # generation deadline must still stop response consumption.
                    for char in body:
                        self.wfile.write(bytes([char]))
                        self.wfile.flush()
                        time.sleep(0.01)
                elif state.mode == "held_stream":
                    first, _ = body.split(b"\n\n", 1)
                    self.wfile.write(first + b"\n\n")
                    self.wfile.flush()
                    self.connection.settimeout(2)
                    if self.connection.recv(1) == b"":
                        state.disconnected.set()
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                state.disconnected.set()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state.origin = f"http://127.0.0.1:{server.server_port}"
    original = socket.socket.connect

    def loopback_only(sock, address):
        assert address[0] in {"127.0.0.1", "::1"}
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        monkeypatch.setenv(key, "http://127.0.0.1:1")
        monkeypatch.setenv(key.lower(), "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:1/forbidden")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/forbidden")
    monkeypatch.setenv("OPENAI_API_KEY", "forbidden-ambient-key")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "forbidden-ambient-key")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "forbidden-ambient-token")
    monkeypatch.setenv("AZURE_API_KEY", "forbidden-ambient-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "http://127.0.0.1:1/forbidden")
    monkeypatch.setenv("AZURE_API_BASE", "http://127.0.0.1:1/forbidden")
    monkeypatch.setenv("OPENAI_API_VERSION", "forbidden-ambient-version")
    monkeypatch.setattr(settings, "PROVIDER_ALLOWED_ORIGINS", [state.origin])
    if state.azure:
        config = {
            "azure_endpoint": state.origin + "/gateway",
            "engine": "fixture-deployment",
            "api_version": "2024-10-21",
            "timeout": 2,
        }
        state.expected_path = "/gateway/openai/deployments/fixture-deployment/chat/completions?api-version=2024-10-21"
        state.expected_auth = (None, "explicit-fixture-key")
        state.wire_model = "fixture-deployment"
    else:
        config = {"api_base": state.origin + "/v1", "timeout": 2}
        state.expected_path = "/v1/chat/completions"
        state.expected_auth = ("Bearer explicit-fixture-key", None)
        state.wire_model = "gpt-4o-mini"
    llama = resolve_llm(
        request.param,
        "gpt-4o-mini",
        config,
        "explicit-fixture-key",
    )
    state.llama = llama
    state.lm = get_dspy_lm_by_llama_llm(llama)
    try:
        yield state
    finally:
        state.lm.close()
        asyncio.run(state.lm.aclose())
        llama._http_client.close()
        asyncio.run(llama._async_http_client.aclose())
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def invoke(lm, asynchronous):
    if asynchronous:
        return asyncio.run(lm.acall(prompt="fixture"))
    return lm(prompt="fixture")


def scope(seconds, attempts=16):
    deadline = time.monotonic() + seconds

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise GenerationExpired("Generation deadline exceeded")
        return value

    result = GenerationScope(None, None)
    result.lease = SimpleNamespace(
        remaining=remaining,
        limiter=SimpleNamespace(
            limits=SimpleNamespace(provider_calls_per_request=attempts)
        ),
    )
    return result


@pytest.mark.parametrize("asynchronous", [False, True])
def test_dspy_real_calls_use_explicit_destination_and_key_without_proxy(
    provider, asynchronous
):
    budget = scope(10)
    token = current_generation.set(budget)
    try:
        assert invoke(provider.lm, asynchronous) == ["fixture answer"]
    finally:
        current_generation.reset(token)
    assert budget.provider_calls == 1
    assert [path for path, _ in provider.calls] == [provider.expected_path]
    assert provider.auth == [provider.expected_auth]
    assert provider.bodies[0]["model"] == provider.wire_model
    if provider.azure:
        assert provider.bodies[0]["max_tokens"] == 4096
        assert "max_completion_tokens" not in provider.bodies[0]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_primary_client_never_substitutes_ambient_azure_identity(
    provider, asynchronous
):
    from llama_index.core.base.llms.types import ChatMessage

    messages = [ChatMessage(role="user", content="fixture")]
    response = (
        asyncio.run(provider.llama.achat(messages))
        if asynchronous
        else provider.llama.chat(messages)
    )
    assert response.message.content == "fixture answer"
    assert provider.auth == [provider.expected_auth]
    assert [path for path, _ in provider.calls] == [provider.expected_path]
    assert provider.bodies[0]["model"] == provider.wire_model


@pytest.mark.parametrize("asynchronous", [False, True])
def test_dspy_custom_engine_preserves_json_adapter_capabilities(provider, asynchronous):
    provider.mode = "structured"
    predictor = dspy.Predict("question -> answer")
    with dspy.context(lm=provider.lm, adapter=dspy.JSONAdapter()):
        result = (
            asyncio.run(predictor.acall(question="fixture"))
            if asynchronous
            else predictor(question="fixture")
        )
    assert result.answer == "fixture answer"
    assert len(provider.calls) == 1
    format_ = provider.bodies[0]["response_format"]
    assert format_["type"] == "json_schema"
    assert format_["json_schema"]["schema"]["properties"]["answer"]["type"] == "string"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["oversized", "compressed", "redirect"])
def test_dspy_standalone_responses_are_bounded(provider, asynchronous, mode):
    provider.mode = mode
    with pytest.raises(LMUnexpectedError) as failure:
        invoke(provider.lm, asynchronous)
    assert isinstance(
        failure.value.__cause__,
        ProviderConfigurationError if mode == "redirect" else GenerationExpired,
    )
    assert len(provider.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_dspy_expired_and_exhausted_budgets_make_no_extra_request(
    provider, asynchronous
):
    token = current_generation.set(scope(-1))
    try:
        with pytest.raises(LMUnexpectedError) as failure:
            invoke(provider.lm, asynchronous)
        assert isinstance(failure.value.__cause__, GenerationExpired)
        assert not provider.calls
    finally:
        current_generation.reset(token)
    token = current_generation.set(scope(10, attempts=1))
    try:
        assert invoke(provider.lm, asynchronous) == ["fixture answer"]
        with pytest.raises(LMUnexpectedError) as failure:
            invoke(provider.lm, asynchronous)
        assert isinstance(failure.value.__cause__, GenerationExpired)
        assert len(provider.calls) == 1
    finally:
        current_generation.reset(token)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_dspy_trickled_body_honors_active_generation_deadline(provider, asynchronous):
    provider.mode = "slow"
    token = current_generation.set(scope(0.15))
    started = time.monotonic()
    try:
        with pytest.raises(LMUnexpectedError) as failure:
            invoke(provider.lm, asynchronous)
        assert isinstance(failure.value.__cause__, GenerationExpired)
        assert time.monotonic() - started < 1
        assert len(provider.calls) == 1
    finally:
        current_generation.reset(token)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_dspy_oversized_request_never_reaches_provider(provider, asynchronous):
    with pytest.raises(LMUnexpectedError) as failure:
        if asynchronous:
            asyncio.run(provider.lm.acall(prompt="x" * (2 * 1024 * 1024)))
        else:
            provider.lm(prompt="x" * (2 * 1024 * 1024))
    assert isinstance(failure.value.__cause__, GenerationExpired)
    assert not provider.calls


def consume_stream(provider, asynchronous, *, early=False):
    request = Request(model=provider.lm.model, messages=(Message.user("fixture"),))
    if asynchronous:

        async def consume():
            events = provider.lm._async_engine_spec.stream(request)
            result = []
            try:
                async for event in events:
                    result.append(event)
                    if early and event.type == "delta":
                        break
            finally:
                await events.aclose()
            return result

        return asyncio.run(consume())
    events = provider.lm.engine.stream(request)
    result = []
    try:
        for event in events:
            result.append(event)
            if early and event.type == "delta":
                break
    finally:
        events.close()
    return result


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_stream_preserves_text_usage_and_closes_owned_socket(
    provider, asynchronous
):
    events = consume_stream(provider, asynchronous)
    assert (
        "".join(
            event.delta.text
            for event in events
            if event.type == "delta" and isinstance(event.delta, TextDelta)
        )
        == "fixture answer"
    )
    assert events[-1].type == "end"
    assert events[-1].usage.input_tokens == 1
    assert events[-1].usage.output_tokens == 2
    provider.mode = "held_stream"
    events = consume_stream(provider, asynchronous, early=True)
    assert any(event.type == "delta" for event in events)
    assert provider.disconnected.wait(1)
    assert provider.auth == [provider.expected_auth] * 2
    assert [path for path, _ in provider.calls] == [provider.expected_path] * 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["oversized", "slow"])
def test_native_stream_keeps_body_and_generation_time_limits(
    provider, asynchronous, mode
):
    provider.mode = mode
    token = current_generation.set(scope(0.15 if mode == "slow" else 10))
    try:
        with pytest.raises(GenerationExpired):
            consume_stream(provider, asynchronous)
        assert len(provider.calls) == 1
    finally:
        current_generation.reset(token)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_primary_sdk_copy_cannot_restore_ambient_identity(provider, asynchronous):
    options = dict(
        model=provider.wire_model, messages=[{"role": "user", "content": "fixture"}]
    )
    if asynchronous:

        async def call():
            client = provider.llama._get_aclient().with_options(timeout=2)
            return await client.chat.completions.create(**options)

        response = asyncio.run(call())
    else:
        client = provider.llama._get_client().with_options(timeout=2)
        response = client.chat.completions.create(**options)
    assert response.choices[0].message.content == "fixture answer"
    assert provider.auth == [provider.expected_auth]
    assert [path for path, _ in provider.calls] == [provider.expected_path]


@pytest.mark.parametrize("provider", [LLMProvider.AZURE_OPENAI], indirect=True)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_azure_keeps_distinct_deployments_versions_and_keys(provider, asynchronous):
    from llama_index.core.base.llms.types import ChatMessage

    other = resolve_llm(
        LLMProvider.AZURE_OPENAI,
        "gpt-4o-mini",
        {
            "azure_endpoint": provider.origin + "/other-gateway/",
            "deployment_name": "other-deployment",
            "api_version": "2025-01-01-preview",
            "timeout": 2,
        },
        "second-explicit-key",
    )
    lm = get_dspy_lm_by_llama_llm(other)
    try:
        messages = [ChatMessage(role="user", content="fixture")]
        primary = (
            asyncio.run(other.achat(messages)) if asynchronous else other.chat(messages)
        )
        assert primary.message.content == "fixture answer"
        assert invoke(lm, asynchronous) == ["fixture answer"]
        assert provider.auth == [(None, "second-explicit-key")] * 2
        assert [path for path, _ in provider.calls] == [
            "/other-gateway/openai/deployments/other-deployment/chat/completions?api-version=2025-01-01-preview"
        ] * 2
        assert [body["model"] for body in provider.bodies] == ["other-deployment"] * 2
        assert invoke(provider.lm, asynchronous) == ["fixture answer"]
        assert provider.auth[-1] == provider.expected_auth
        assert provider.calls[-1][0] == provider.expected_path
    finally:
        lm.close()
        asyncio.run(lm.aclose())
        other._http_client.close()
        asyncio.run(other._async_http_client.aclose())


def test_native_adapter_preserves_tools_and_conversation(provider):
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}},
        },
    }
    messages = [
        {"role": "user", "content": "fixture"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "fixture-call",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id":7}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "fixture-call", "content": "fixture result"},
    ]
    assert provider.lm(messages=messages, tools=[tool], max_tokens=200) == [
        "fixture answer"
    ]
    body = provider.bodies[0]
    assert body["tools"][0]["function"]["name"] == "lookup"
    assert json.loads(
        body["messages"][1]["tool_calls"][0]["function"]["arguments"]
    ) == {"id": 7}
    assert body["messages"][2]["tool_call_id"] == "fixture-call"
    assert body["model"] == provider.wire_model
    assert provider.auth == [provider.expected_auth]


@pytest.mark.parametrize(
    "model,limit,token_field",
    [("gpt-4o-mini", 4096, "max_tokens"), ("o3-mini", 8192, "max_completion_tokens")],
)
def test_primary_and_dspy_keep_model_specific_token_parameter(
    provider, model, limit, token_field
):
    from llama_index.core.base.llms.types import ChatMessage

    config = {"max_tokens": limit, "temperature": 1, "timeout": 2}
    if provider.azure:
        config.update(
            azure_endpoint=provider.origin,
            engine="model-fixture",
            api_version="2025-01-01-preview"
            if model == "o3-mini"
            else "2024-02-15-preview",
        )
    else:
        config["api_base"] = provider.origin + "/v1"
    primary = resolve_llm(
        LLMProvider.AZURE_OPENAI if provider.azure else LLMProvider.OPENAI,
        model,
        config,
        "model-fixture-key",
    )
    lm = get_dspy_lm_by_llama_llm(primary)
    try:
        assert (
            primary.chat([ChatMessage(role="user", content="fixture")]).message.content
            == "fixture answer"
        )
        assert lm(prompt="fixture") == ["fixture answer"]
        for body in provider.bodies:
            assert body[token_field] == limit
            assert (set(body) & {"max_tokens", "max_completion_tokens"}) == {
                token_field
            }
        assert lm(prompt="fixture", max_tokens=16000) == ["fixture answer"]
        assert provider.bodies[-1][token_field] == limit
        assert lm(prompt="fixture", max_tokens=128) == ["fixture answer"]
        assert provider.bodies[-1][token_field] == 128
        copied = lm.copy()
        try:
            assert copied(prompt="fixture") == ["fixture answer"]
            assert provider.bodies[-1][token_field] == limit
        finally:
            copied.close()
    finally:
        lm.close()
        primary._http_client.close()
        asyncio.run(primary._async_http_client.aclose())
