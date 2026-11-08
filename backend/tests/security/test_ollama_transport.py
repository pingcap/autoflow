"""Exercise actual Ollama HTTP, including DSPy's JSON and streaming adapters."""

import asyncio
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace

import dspy
import litellm
import pytest
from dspy.lm15 import Message, Request, TextDelta, ToolCallPart
from dspy.utils.exceptions import LMUnexpectedError
from llama_index.core.base.llms.types import ChatMessage
from ollama import ResponseError

from app.core.config import settings
from app.rag.llms.dspy import get_dspy_lm_by_llama_llm
from app.rag.llms.provider import LLMProvider
from app.rag.llms.resolver import resolve_llm
from app.security.generation import GenerationScope, current_generation
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError


@pytest.fixture
def provider(monkeypatch):
    state = SimpleNamespace(calls=[], mode="normal", disconnected=Event())

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append((self.path, self.headers.get("Authorization"), request))
            content = "fixture answer"
            if state.mode == "oversized":
                content = "x" * (4 * 1024 * 1024 + 1)
            if state.mode == "structured":
                content = '{"answer": "fixture answer"}'
            message = {"role": "assistant", "content": content}
            if state.mode == "tool":
                message = {
                    "role": "assistant",
                    "content": "",
                    "thinking": "fixture thought",
                    "tool_calls": [
                        {"function": {"name": "lookup", "arguments": {"id": 7}}}
                    ],
                }
            chunk = {
                "model": "llama3.2",
                "created_at": "2026-01-01T00:00:00Z",
                "message": message,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 2,
            }
            if request.get("stream"):
                first = dict(
                    chunk, done=False, message=dict(message, content="fixture ")
                )
                if state.mode == "oversized":
                    first["message"]["content"] = content
                last = dict(chunk, message={"role": "assistant", "content": "answer"})
                if state.mode == "tool":
                    first["message"] = message
                    last["message"]["content"] = ""
                body = json.dumps(first).encode() + b"\n"
                if state.mode != "truncated":
                    body += json.dumps(last).encode() + b"\n"
            else:
                body = json.dumps(chunk).encode()
            if state.mode == "error":
                body = b'{"error":"private upstream body fixture"}'
            if state.mode == "malformed":
                body = b'{"message":"private upstream body fixture"}'
            status = (
                302
                if state.mode == "redirect"
                else 400
                if state.mode == "error"
                else 200
            )
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/x-ndjson" if request.get("stream") else "application/json",
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            if state.mode == "compressed":
                self.send_header("Content-Encoding", "gzip")
            if state.mode == "redirect":
                self.send_header("Location", state.origin + "/forbidden")
            self.end_headers()
            try:
                if state.mode == "slow":
                    for char in body:
                        self.wfile.write(bytes([char]))
                        self.wfile.flush()
                        time.sleep(0.01)
                elif state.mode == "held_stream":
                    first, rest = body.split(b"\n", 1)
                    self.wfile.write(first + b"\n")
                    self.wfile.flush()
                    # Deterministic early close evidence, without sending more data.
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

    def local_only(sock, address):
        assert address[0] in {"127.0.0.1", "::1"}
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
    for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        monkeypatch.setenv(name, "http://127.0.0.1:1")
        monkeypatch.setenv(name.lower(), "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("OLLAMA_API_KEY", "forbidden-ambient-key")
    monkeypatch.setenv("OLLAMA_API_BASE", "http://127.0.0.1:1/forbidden")
    monkeypatch.setattr(litellm, "api_key", "forbidden-global-key")
    monkeypatch.setattr(litellm.OllamaChatConfig, "num_predict", 999999)
    monkeypatch.setattr(settings, "PROVIDER_ALLOWED_ORIGINS", [state.origin])
    state.llama = resolve_llm(
        LLMProvider.OLLAMA,
        "llama3.2",
        {"base_url": state.origin, "request_timeout": 2},
        "explicit-fixture-key",
    )
    state.lm = get_dspy_lm_by_llama_llm(state.llama)
    try:
        yield state
    finally:
        state.lm.close()
        asyncio.run(state.lm.aclose())
        state.llama.client._client.close()
        asyncio.run(state.llama.async_client._client.aclose())
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def invoke(provider, client, asynchronous, prompt="fixture"):
    if client == "dspy":
        return (
            asyncio.run(provider.lm.acall(prompt=prompt))
            if asynchronous
            else provider.lm(prompt=prompt)
        )[0]
    messages = [ChatMessage(role="user", content=prompt)]
    response = (
        asyncio.run(provider.llama.achat(messages))
        if asynchronous
        else provider.llama.chat(messages)
    )
    return response.message.content


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
@pytest.mark.parametrize("client", ["llama", "dspy"])
def test_ollama_explicit_binding_and_one_budgeted_request(
    provider, client, asynchronous
):
    budget = scope(10)
    token = current_generation.set(budget)
    try:
        assert invoke(provider, client, asynchronous) == "fixture answer"
    finally:
        current_generation.reset(token)
    assert budget.provider_calls == 1
    assert [(path, key) for path, key, _ in provider.calls] == [
        ("/api/chat", "Bearer explicit-fixture-key")
    ]
    assert provider.calls[0][2]["options"].get("num_predict") != 999999


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("client", ["llama", "dspy"])
@pytest.mark.parametrize("mode", ["oversized", "compressed", "redirect"])
def test_ollama_responses_are_bounded_without_chat_lease(
    provider, client, asynchronous, mode
):
    provider.mode = mode
    with pytest.raises(
        (
            GenerationExpired,
            ProviderConfigurationError,
            LMUnexpectedError,
            ResponseError,
        )
    ) as failure:
        invoke(provider, client, asynchronous)
    error = failure.value.__cause__ if client == "dspy" else failure.value
    assert isinstance(
        error,
        (ProviderConfigurationError, ResponseError)
        if mode == "redirect"
        else GenerationExpired,
    )
    assert len(provider.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("client", ["llama", "dspy"])
def test_ollama_request_size_and_expired_or_exhausted_budgets(
    provider, client, asynchronous
):
    expected = LMUnexpectedError if client == "dspy" else GenerationExpired
    with pytest.raises(expected):
        invoke(provider, client, asynchronous, "x" * (2 * 1024 * 1024))
    assert not provider.calls
    token = current_generation.set(scope(-1))
    try:
        with pytest.raises(expected):
            invoke(provider, client, asynchronous)
        assert not provider.calls
    finally:
        current_generation.reset(token)
    token = current_generation.set(scope(10, attempts=1))
    try:
        assert invoke(provider, client, asynchronous) == "fixture answer"
        with pytest.raises(expected):
            invoke(provider, client, asynchronous)
        assert len(provider.calls) == 1
    finally:
        current_generation.reset(token)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("client", ["llama", "dspy"])
def test_ollama_trickle_cannot_extend_generation_deadline(
    provider, client, asynchronous
):
    provider.mode = "slow"
    token = current_generation.set(scope(0.15))
    started = time.monotonic()
    try:
        with pytest.raises(
            LMUnexpectedError if client == "dspy" else GenerationExpired
        ) as failure:
            invoke(provider, client, asynchronous)
        error = failure.value.__cause__ if client == "dspy" else failure.value
        assert isinstance(error, GenerationExpired)
        assert time.monotonic() - started < 1
        assert len(provider.calls) == 1
    finally:
        current_generation.reset(token)


def native_request():
    return Request(model="ollama_chat/llama3.2", messages=(Message.user("fixture"),))


def stream(provider, asynchronous, early=False):
    if asynchronous:

        async def consume():
            events = provider.lm._async_engine_spec.stream(native_request())
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
    events = provider.lm.engine.stream(native_request())
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
def test_ollama_stream_preserves_text_usage_and_closes_early(provider, asynchronous):
    events = stream(provider, asynchronous)
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
    events = stream(provider, asynchronous, early=True)
    assert any(event.type == "delta" for event in events)
    assert provider.disconnected.wait(1)
    assert len(provider.calls) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_ollama_truncated_stream_is_not_a_success(provider, asynchronous):
    provider.mode = "truncated"
    with pytest.raises(ProviderConfigurationError, match="Incomplete"):
        stream(provider, asynchronous)


@pytest.mark.parametrize("mode", ["error", "malformed"])
def test_ollama_dspy_errors_do_not_echo_upstream_body(provider, mode):
    provider.mode = mode
    with pytest.raises(LMUnexpectedError) as failure:
        invoke(provider, "dspy", False)
    assert isinstance(failure.value.__cause__, ProviderConfigurationError)
    assert "private upstream body" not in str(failure.value) + str(
        failure.value.__cause__
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_ollama_dspy_json_adapter_preserves_schema(provider, asynchronous):
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
    body = provider.calls[0][2]
    assert body["format"]["properties"]["answer"]["type"] == "string"
    assert body["options"]["num_predict"] > 0


@pytest.mark.parametrize("asynchronous", [False, True])
def test_ollama_native_tools_and_thinking(provider, asynchronous):
    provider.mode = "tool"
    if asynchronous:
        response = asyncio.run(
            provider.lm._async_engine_spec.complete(native_request())
        )
    else:
        response = provider.lm.engine.complete(native_request())
    tool = response.message.parts_of(ToolCallPart)[0]
    assert tool.id and tool.name == "lookup" and tool.input == {"id": 7}
    assert response.finish_reason == "tool_call"
    events = stream(provider, asynchronous)
    assert events[-1].finish_reason == "tool_call"
    assert any(
        event.type == "delta" and event.delta.type == "thinking" for event in events
    )
    assert len(provider.calls) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["oversized", "compressed", "slow"])
def test_ollama_stream_uses_same_response_and_time_budgets(
    provider, asynchronous, mode
):
    provider.mode = mode
    token = current_generation.set(scope(0.15 if mode == "slow" else 10))
    try:
        with pytest.raises(GenerationExpired):
            stream(provider, asynchronous)
        assert len(provider.calls) == 1
    finally:
        current_generation.reset(token)


def test_ollama_dspy_preserves_inline_images_and_tool_history(provider):
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}},
        },
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "fixture"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "fixture-call",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id":7}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "fixture-call", "content": "fixture result"},
    ]
    assert provider.lm(messages=messages, tools=[tool], top_k=17, max_tokens=200) == [
        "fixture answer"
    ]
    body = provider.calls[0][2]
    assert body["messages"][0]["images"] == ["aW1hZ2U="]
    assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"id": 7}
    assert body["messages"][2]["tool_call_id"] == "fixture-call"
    assert body["tools"][0]["function"]["name"] == "lookup"
    assert body["options"]["num_predict"] == 200
    assert body["options"]["top_k"] == 17
    assert len(provider.calls) == 1


def test_ollama_copies_keys_and_event_loops_remain_isolated(provider):
    from concurrent.futures import ThreadPoolExecutor

    other = resolve_llm(
        LLMProvider.OLLAMA,
        "llama3.2",
        {"base_url": provider.origin, "request_timeout": 2},
        "second-fixture-key",
    )
    other_lm = get_dspy_lm_by_llama_llm(other)
    copied = provider.lm.copy(temperature=0.2)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(lm, prompt="fixture") for lm in [copied, other_lm]]
            assert [task.result() for task in tasks] == [
                ["fixture answer"],
                ["fixture answer"],
            ]
        assert sorted(key for _, key, _ in provider.calls) == [
            "Bearer explicit-fixture-key",
            "Bearer second-fixture-key",
        ]
        for _ in range(2):
            assert asyncio.run(provider.lm.acall(prompt="fixture")) == [
                "fixture answer"
            ]
        assert len(provider.calls) == 4
    finally:
        copied.close()
        other_lm.close()
        other.client._client.close()
        asyncio.run(other.async_client._client.aclose())
