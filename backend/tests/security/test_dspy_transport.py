import asyncio
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest
import dspy
from dspy.utils.exceptions import LMUnexpectedError

from app.core.config import settings
from app.rag.llms.dspy import get_dspy_lm_by_llama_llm
from app.rag.llms.provider import LLMProvider
from app.rag.llms.resolver import resolve_llm
from app.security.generation import GenerationScope, current_generation
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError


@pytest.fixture
def provider(monkeypatch):
    state = SimpleNamespace(calls=[], bodies=[], mode="normal")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            state.bodies.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            state.calls.append((self.path, self.headers.get("Authorization")))
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
            self.send_response(302 if state.mode == "redirect" else 200)
            self.send_header("Content-Type", "application/json")
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
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

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
    monkeypatch.setattr(settings, "PROVIDER_ALLOWED_ORIGINS", [state.origin])
    llama = resolve_llm(
        LLMProvider.OPENAI,
        "gpt-4o-mini",
        {"api_base": state.origin + "/v1", "timeout": 2},
        "explicit-fixture-key",
    )
    state.lm = get_dspy_lm_by_llama_llm(llama)
    try:
        yield state
    finally:
        state.lm.close()
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
    assert provider.calls == [("/v1/chat/completions", "Bearer explicit-fixture-key")]


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
