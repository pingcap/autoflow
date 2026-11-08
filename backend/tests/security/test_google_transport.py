"""Real Google SDK/native DSPy traffic, using only owned loopback fixtures."""

import asyncio
import base64
import datetime
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace
from urllib.parse import parse_qs

import dspy
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from dspy.lm15 import Message, Request, TextDelta
from dspy.utils.exceptions import LMUnexpectedError

from app.core.config import settings
from app.rag.llms.dspy import get_dspy_lm_by_llama_llm
from app.rag.llms.provider import LLMProvider
from app.rag.llms.resolver import resolve_llm
from app.security import google_http
from app.security.generation import GenerationScope, current_generation
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError


@pytest.fixture(scope="module")
def service_account():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return {
        "type": "service_account",
        "project_id": "fixture-project",
        "private_key_id": "fixture-id",
        "private_key": private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode(),
        "client_email": "fixture@fixture-project.iam.gserviceaccount.com",
        "token_uri": "https://oauth2.googleapis.com/token",
    }


@pytest.fixture(params=[LLMProvider.GEMINI, LLMProvider.VERTEX_AI])
def provider(monkeypatch, request, service_account, tmp_path):
    state = SimpleNamespace(
        calls=[], bodies=[], auth=[], oauth=[], mode="normal", disconnected=Event()
    )
    state.vertex = request.param == LLMProvider.VERTEX_AI

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send_body(self, body, *, streaming=False, mode=None):
            mode = mode or state.mode
            self.send_response(302 if mode == "redirect" else 200)
            self.send_header(
                "Content-Type", "text/event-stream" if streaming else "application/json"
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            if mode == "compressed":
                self.send_header("Content-Encoding", "gzip")
            if mode == "redirect":
                self.send_header("Location", state.origin + "/should-not-follow")
            self.end_headers()
            try:
                if mode == "slow":
                    for char in body:
                        self.wfile.write(bytes([char]))
                        self.wfile.flush()
                        time.sleep(0.01)
                elif mode == "held_stream":
                    self.wfile.write(body.split(b"\n\n", 1)[0] + b"\n\n")
                    self.wfile.flush()
                    self.connection.settimeout(2)
                    if self.connection.recv(1) == b"":
                        state.disconnected.set()
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                state.disconnected.set()

        def record(self):
            state.calls.append((self.command, self.path))
            state.auth.append(
                (self.headers.get("x-goog-api-key"), self.headers.get("Authorization"))
            )

        def do_GET(self):
            self.record()
            payload = {
                "name": "models/gemini-2.5-flash",
                "inputTokenLimit": 1000000,
                "outputTokenLimit": 8192,
            }
            if state.mode == "oversized":
                payload["description"] = "x" * (4 * 1024 * 1024 + 1)
            self.send_body(json.dumps(payload).encode())

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/token":
                form = parse_qs(body.decode())
                assertion = form["assertion"][0].split(".")[1]
                claims = json.loads(
                    base64.urlsafe_b64decode(assertion + "=" * (-len(assertion) % 4))
                )
                state.oauth.append(claims)
                token = f"fixture-access-{len(state.oauth)}"
                payload = {
                    "access_token": token,
                    "expires_in": 3600,
                    "token_type": "Bearer",
                }
                if state.mode == "oversized":
                    payload["padding"] = "x" * (4 * 1024 * 1024 + 1)
                self.send_body(json.dumps(payload).encode())
                return
            self.record()
            state.bodies.append(json.loads(body))
            content = "fixture answer"
            if state.mode == "structured":
                content = '{"answer": "fixture answer"}'
            elif state.mode == "oversized":
                content = "x" * (4 * 1024 * 1024 + 1)
            data = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": content}]},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 3,
                },
                "modelVersion": "gemini-2.5-flash",
            }
            streaming = ":streamGenerateContent" in self.path
            if streaming:
                first = json.loads(json.dumps(data))
                first["candidates"][0].pop("finishReason")
                first.pop("usageMetadata")
                data["candidates"][0]["content"]["parts"] = [{"text": ""}]
                body = (
                    "data: "
                    + json.dumps(first)
                    + "\n\ndata: "
                    + json.dumps(data)
                    + "\n\n"
                ).encode()
            else:
                body = json.dumps(data).encode()
            self.send_body(body, streaming=streaming)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state.origin = f"http://127.0.0.1:{server.server_port}"
    original_connect = socket.socket.connect

    def owned_server_only(sock, address):
        assert address == ("127.0.0.1", server.server_port)
        return original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", owned_server_only)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
        monkeypatch.setenv(key.lower(), "http://127.0.0.1:1")
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(key, "")
    for key in ("GOOGLE_GEMINI_BASE_URL", "GOOGLE_VERTEX_BASE_URL"):
        monkeypatch.setenv(key, "http://127.0.0.1:1/forbidden")
    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(key, "forbidden-ambient-key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "forbidden-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "forbidden-location")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("GOOGLE_GENAI_CLIENT_MODE", "record")
    monkeypatch.setenv(
        "GOOGLE_GENAI_REPLAYS_DIRECTORY", str(tmp_path / "forbidden-recordings")
    )
    monkeypatch.setattr(settings, "PROVIDER_ALLOWED_ORIGINS", [])
    monkeypatch.setattr(google_http, "GEMINI_ENDPOINT", state.origin)
    monkeypatch.setattr(google_http, "vertex_endpoint", lambda _location: state.origin)
    monkeypatch.setattr(google_http, "TOKEN_ENDPOINT", state.origin + "/token")

    class FixtureServiceAccount(google_http.BoundServiceAccount):
        def __init__(self, *args, **kwargs):
            # Keep real config validation, signing and OAuth exchange. Only the
            # test-owned endpoint replaces Google's standard token destination.
            kwargs["token_uri"] = state.origin + "/token"
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(google_http, "BoundServiceAccount", FixtureServiceAccount)
    state.model = "gemini-2.5-flash" if state.vertex else "models/gemini-2.5-flash"
    state.config = {"timeout": 2, "max_tokens": 128, "max_retries": 0}
    state.credentials = service_account if state.vertex else "explicit-fixture-key"
    if state.vertex:
        state.config["location"] = "us-west1"
    state.construct = lambda: resolve_llm(
        request.param, state.model, state.config, state.credentials
    )
    state.llama = state.construct()
    state.lm = get_dspy_lm_by_llama_llm(state.llama)
    state.expected_auth = (
        (None, "Bearer fixture-access-1")
        if state.vertex
        else ("explicit-fixture-key", None)
    )
    state.path = (
        "/v1beta1/projects/fixture-project/locations/us-west1/publishers/google/models/gemini-2.5-flash"
        if state.vertex
        else "/v1beta/models/gemini-2.5-flash"
    )
    state.metadata_calls = state.calls[:]
    state.metadata_auth = state.auth[:]
    state.calls.clear()
    state.auth.clear()
    try:
        yield state
    finally:
        state.lm.close()
        asyncio.run(state.lm.aclose())
        state.llama._client.close()
        asyncio.run(state.llama._client.aio.aclose())
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
        assert not (tmp_path / "forbidden-recordings").exists()


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


def invoke(provider, asynchronous, client, **kwargs):
    if client == "dspy":
        return (
            asyncio.run(provider.lm.acall(prompt="fixture", **kwargs))
            if asynchronous
            else provider.lm(prompt="fixture", **kwargs)
        )
    result = (
        asyncio.run(provider.llama.acomplete("fixture", **kwargs))
        if asynchronous
        else provider.llama.complete("fixture", **kwargs)
    )
    return [result.text]


@pytest.mark.parametrize("client", ["primary", "dspy"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_google_requests_bind_identity_destination_and_output_limit(
    provider, asynchronous, client
):
    budget = scope(10)
    token = current_generation.set(budget)
    try:
        assert invoke(provider, asynchronous, client) == ["fixture answer"]
        override = (
            {"max_tokens": 8192}
            if client == "dspy"
            else {"generation_config": {"max_output_tokens": 8192}}
        )
        assert invoke(provider, asynchronous, client, **override) == ["fixture answer"]
        override = (
            {"max_tokens": 64}
            if client == "dspy"
            else {"generation_config": {"max_output_tokens": 64}}
        )
        assert invoke(provider, asynchronous, client, **override) == ["fixture answer"]
    finally:
        current_generation.reset(token)
    assert budget.provider_calls == 3
    assert provider.calls == [("POST", provider.path + ":generateContent")] * 3
    assert provider.auth == [provider.expected_auth] * 3
    assert [
        body["generationConfig"]["maxOutputTokens"] for body in provider.bodies
    ] == [128, 128, 64]
    assert provider.llama._client._api_client._use_aiohttp() is False
    assert provider.metadata_auth == [provider.expected_auth]
    assert len(provider.metadata_calls) == 1
    assert provider.metadata_calls[0][0] == "GET"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_google_json_adapter_preserves_schema(provider, asynchronous):
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
    config = provider.bodies[0]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    schema = config.get("responseJsonSchema") or config.get("responseSchema")
    assert schema["properties"]["answer"]["type"] == "string"


@pytest.mark.parametrize("mode", ["oversized", "compressed", "redirect"])
@pytest.mark.parametrize("client", ["primary", "dspy"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_google_response_limits(provider, asynchronous, client, mode):
    provider.mode = mode
    expected = ProviderConfigurationError if mode == "redirect" else GenerationExpired
    with pytest.raises(LMUnexpectedError if client == "dspy" else expected) as failure:
        invoke(provider, asynchronous, client)
    if client == "dspy":
        assert isinstance(failure.value.__cause__, expected)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("client", ["primary", "dspy"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_google_expiry_attempts_and_trickled_response(provider, asynchronous, client):
    expected = LMUnexpectedError if client == "dspy" else GenerationExpired
    token = current_generation.set(scope(-1))
    try:
        with pytest.raises(expected):
            invoke(provider, asynchronous, client)
        assert not provider.calls
    finally:
        current_generation.reset(token)
    token = current_generation.set(scope(10, attempts=1))
    try:
        assert invoke(provider, asynchronous, client) == ["fixture answer"]
        with pytest.raises(expected):
            invoke(provider, asynchronous, client)
        assert len(provider.calls) == 1
    finally:
        current_generation.reset(token)
    provider.mode = "slow"
    token = current_generation.set(scope(0.15))
    started = time.monotonic()
    try:
        with pytest.raises(expected):
            invoke(provider, asynchronous, client)
        assert time.monotonic() - started < 1
        assert len(provider.calls) == 2
    finally:
        current_generation.reset(token)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("client", ["primary", "dspy"])
def test_google_identity_survives_expiry_without_ambient_credentials(
    provider, asynchronous, client
):
    if provider.vertex:
        credentials = provider.llama._client._api_client._credentials
        credentials.expiry = datetime.datetime.now(datetime.timezone.utc).replace(
            tzinfo=None
        ) - datetime.timedelta(seconds=10)
    budget = scope(10)
    token = current_generation.set(budget)
    try:
        assert invoke(provider, asynchronous, client) == ["fixture answer"]
    finally:
        current_generation.reset(token)
    if provider.vertex:
        assert len(provider.oauth) == 2
        assert budget.provider_calls == 2  # OAuth renewal and model generation.
        assert provider.auth == [(None, "Bearer fixture-access-2")]
        assert (
            provider.oauth[-1]["iss"]
            == "fixture@fixture-project.iam.gserviceaccount.com"
        )
        assert (
            provider.oauth[-1]["scope"]
            == "https://www.googleapis.com/auth/cloud-platform"
        )
    else:
        assert not provider.oauth
        assert budget.provider_calls == 1
        assert provider.auth == [provider.expected_auth]


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
def test_google_native_stream_and_early_socket_close(provider, asynchronous):
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
    assert any(
        event.type == "delta"
        for event in consume_stream(provider, asynchronous, early=True)
    )
    assert provider.disconnected.wait(1)
    assert (
        provider.calls
        == [("POST", provider.path + ":streamGenerateContent?alt=sse")] * 2
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["oversized", "compressed", "redirect"])
def test_google_metadata_reads_are_bounded(provider, asynchronous, mode):
    provider.mode = mode
    expected = ProviderConfigurationError if mode == "redirect" else GenerationExpired
    with pytest.raises(expected):
        if asynchronous:
            asyncio.run(provider.llama._client.aio.models.get(model=provider.model))
        else:
            provider.llama._client.models.get(model=provider.model)
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == "GET"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("client", ["primary", "dspy"])
def test_google_oversized_requests_never_reach_provider(provider, asynchronous, client):
    prompt = "x" * (2 * 1024 * 1024 + 1)
    with pytest.raises(
        LMUnexpectedError if client == "dspy" else GenerationExpired
    ) as failure:
        if client == "primary":
            if asynchronous:
                asyncio.run(provider.llama.acomplete(prompt))
            else:
                provider.llama.complete(prompt)
        elif asynchronous:
            asyncio.run(provider.lm.acall(prompt=prompt))
        else:
            provider.lm(prompt=prompt)
    if client == "dspy":
        assert isinstance(failure.value.__cause__, GenerationExpired)
    assert not provider.calls


def primary_stream(provider, asynchronous, *, early=False):
    if asynchronous:

        async def consume():
            events = await provider.llama.astream_complete("fixture")
            result = []
            try:
                async for event in events:
                    result.append(event)
                    if early:
                        break
            finally:
                await events.aclose()
            return result

        return asyncio.run(consume())
    events = provider.llama.stream_complete("fixture")
    result = []
    try:
        for event in events:
            result.append(event)
            if early:
                break
    finally:
        events.close()
    return result


@pytest.mark.parametrize("asynchronous", [False, True])
def test_primary_google_stream_returns_answer(provider, asynchronous):
    events = primary_stream(provider, asynchronous)
    assert events[-1].text == "fixture answer"
    assert provider.calls == [
        ("POST", provider.path + ":streamGenerateContent?alt=sse")
    ]
    assert provider.auth == [provider.expected_auth]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_primary_google_early_stream_exit_closes_socket(provider, asynchronous):
    provider.mode = "held_stream"
    assert (
        primary_stream(provider, asynchronous, early=True)[0].text == "fixture answer"
    )
    assert provider.disconnected.wait(1)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("client", ["primary", "dspy"])
@pytest.mark.parametrize("mode", ["oversized", "slow"])
def test_google_stream_body_and_deadline_limits(provider, asynchronous, client, mode):
    provider.mode = mode
    token = current_generation.set(scope(0.15 if mode == "slow" else 10))
    try:
        with pytest.raises(GenerationExpired):
            (primary_stream if client == "primary" else consume_stream)(
                provider, asynchronous
            )
    finally:
        current_generation.reset(token)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_primary_google_per_call_options_cannot_change_destination(
    provider, asynchronous
):
    from google.genai.types import GenerateContentConfig, HttpOptions

    options = GenerateContentConfig(
        http_options=HttpOptions(base_url="http://127.0.0.1:1/forbidden")
    )
    with pytest.raises(ProviderConfigurationError):
        if asynchronous:
            asyncio.run(
                provider.llama._client.aio.models.generate_content(
                    model=provider.model, contents="fixture", config=options
                )
            )
        else:
            provider.llama._client.models.generate_content(
                model=provider.model, contents="fixture", config=options
            )
    assert not provider.calls


@pytest.mark.parametrize("asynchronous", [False, True])
def test_google_tool_history_reaches_native_provider(provider, asynchronous):
    messages = [
        {"role": "user", "content": "lookup fixture"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "fixture-call",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"name":"fixture"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "fixture-call", "content": "fixture result"},
    ]
    result = (
        asyncio.run(provider.lm.acall(messages=messages))
        if asynchronous
        else provider.lm(messages=messages)
    )
    assert result == ["fixture answer"]
    parts = [
        part for content in provider.bodies[0]["contents"] for part in content["parts"]
    ]
    assert any(part.get("functionCall", {}).get("name") == "lookup" for part in parts)
    assert any(
        part.get("functionResponse", {}).get("name") == "lookup" for part in parts
    )


@pytest.mark.parametrize("provider", [LLMProvider.VERTEX_AI], indirect=True)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("client", ["primary", "dspy"])
@pytest.mark.parametrize("mode", ["oversized", "compressed", "redirect", "slow"])
def test_vertex_renewal_failures_are_bounded_before_model_request(
    provider, asynchronous, client, mode
):
    provider.llama._client._api_client._credentials.token = None
    provider.mode = mode
    expected = ProviderConfigurationError if mode == "redirect" else GenerationExpired
    budget = scope(0.15 if mode == "slow" else 10)
    token = current_generation.set(budget)
    try:
        with pytest.raises(
            LMUnexpectedError if client == "dspy" else expected
        ) as failure:
            invoke(provider, asynchronous, client)
        if client == "dspy":
            assert isinstance(failure.value.__cause__, expected)
    finally:
        current_generation.reset(token)
    assert len(provider.oauth) == 2
    assert budget.provider_calls == 1
    assert not provider.calls


@pytest.mark.parametrize("provider", [LLMProvider.VERTEX_AI], indirect=True)
def test_copied_service_accounts_keep_guarded_refresh(provider):
    original = provider.llama._client._api_client._credentials
    copied = original.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])

    def forbidden_request(*_args, **_kwargs):
        pytest.fail("The SDK's unguarded request object was used")

    copied.refresh(forbidden_request)
    assert copied.token == "fixture-access-2"
    assert len(provider.oauth) == 2
    changed = copied.with_token_uri("http://127.0.0.1:1/forbidden")
    with pytest.raises(ProviderConfigurationError):
        changed.refresh(forbidden_request)
    assert len(provider.oauth) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_model_resource_names_and_copied_lms_preserve_google_binding(
    provider, asynchronous
):
    provider.model = (
        "publishers/google/models/gemini-2.5-flash"
        if provider.vertex
        else "gemini-2.5-flash"
    )
    llama = provider.construct()
    lm = get_dspy_lm_by_llama_llm(llama)
    copied = lm.copy(max_tokens=8192)
    provider.mode = "structured"
    provider.calls.clear()
    provider.auth.clear()
    predictor = dspy.Predict("question -> answer")
    try:
        with dspy.context(lm=copied, adapter=dspy.JSONAdapter()):
            result = (
                asyncio.run(predictor.acall(question="fixture"))
                if asynchronous
                else predictor(question="fixture")
            )
        assert result.answer == "fixture answer"
        config = provider.bodies[-1]["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config.get("responseJsonSchema") or config.get("responseSchema")
        assert config["maxOutputTokens"] == 128
        assert provider.calls == [("POST", provider.path + ":generateContent")]
        expected_auth = (
            (None, "Bearer fixture-access-2")
            if provider.vertex
            else provider.expected_auth
        )
        assert provider.auth == [expected_auth]
    finally:
        copied.close()
        asyncio.run(copied.aclose())
        lm.close()
        asyncio.run(lm.aclose())
        llama._client.close()
        asyncio.run(llama._client.aio.aclose())


@pytest.mark.parametrize("provider", [LLMProvider.VERTEX_AI], indirect=True)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("client", ["primary", "dspy"])
def test_expired_vertex_token_cannot_bypass_generation_admission(
    provider, asynchronous, client
):
    provider.llama._client._api_client._credentials.token = None
    expected = LMUnexpectedError if client == "dspy" else GenerationExpired
    token = current_generation.set(scope(-1))
    try:
        with pytest.raises(expected):
            invoke(provider, asynchronous, client)
        assert len(provider.oauth) == 1
        assert not provider.calls
    finally:
        current_generation.reset(token)
    token = current_generation.set(scope(10, attempts=1))
    try:
        with pytest.raises(expected):
            invoke(provider, asynchronous, client)
        assert len(provider.oauth) == 2
        assert not provider.calls
    finally:
        current_generation.reset(token)
