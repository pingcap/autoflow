from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from uuid import uuid4
import json
import time

import pytest
import requests
from fastapi import HTTPException
from pydantic import ValidationError

from app.core.config import settings
from app.rag.chat.config import ChatEngineConfig
from app.security.external_engine import external_engine_session
from app.security.generation import current_generation
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError
from app.security.provider_http import ProviderSession


@contextmanager
def service():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append((self.path, self.headers.get("Authorization"), body))
            self.send_response(302 if self.path == "/redirect" else 200)
            if self.path == "/redirect":
                self.send_header("Location", "/stolen")
            if self.path == "/compressed":
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            try:
                if self.path == "/large":
                    self.wfile.write(b"x" * (4 * 1024 * 1024 + 1))
                elif self.path == "/verify":
                    self.wfile.write(b'{"job_id":"123"}')
                elif self.path == "/slow":
                    self.wfile.write(b"first\n")
                    self.wfile.flush()
                    time.sleep(0.2)
                    self.wfile.write(b"last\n")
                else:
                    self.wfile.write(b'0:"answer"\n8:[{"task_id":"task"}]\n')
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_engine_configuration_and_legacy_runtime_require_operator_approval(monkeypatch):
    from app.repositories.chat_engine import _validate_options

    origin = "https://engine.example.test"
    options = {"external_engine_config": {"stream_chat_api_url": origin + "/chat"}}
    monkeypatch.setattr(settings, "EXTERNAL_ENGINE_ALLOWED_ORIGINS", [])
    with pytest.raises(ValidationError):
        ChatEngineConfig.model_validate(options)
    with pytest.raises(HTTPException) as error:
        _validate_options(options)
    assert error.value.status_code == 400
    assert origin not in error.value.detail
    monkeypatch.setattr(settings, "EXTERNAL_ENGINE_ALLOWED_ORIGINS", [origin])
    assert ChatEngineConfig.model_validate(options).is_external_engine
    for url in [
        "file:///etc/passwd",
        origin + "@attacker.test",
        origin + ".attacker.test",
        origin + "/?key=secret",
    ]:
        with pytest.raises(ValidationError):
            ChatEngineConfig(post_verification_url=url)
    # Runtime rechecks the operator policy, including objects loaded earlier.
    monkeypatch.setattr(settings, "EXTERNAL_ENGINE_ALLOWED_ORIGINS", [])
    with pytest.raises(ProviderConfigurationError):
        external_engine_session(origin + "/chat")


def test_real_stream_preserves_protocol_without_ambient_proxy_or_netrc(
    monkeypatch, tmp_path
):
    with service() as (origin, calls):
        monkeypatch.setattr(settings, "EXTERNAL_ENGINE_ALLOWED_ORIGINS", [origin])
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("NO_PROXY", "")
        netrc = tmp_path / "netrc"
        netrc.write_text("machine 127.0.0.1 login ambient password forbidden\n")
        monkeypatch.setenv("NETRC", str(netrc))
        with external_engine_session(origin) as session:
            with session.post(
                origin + "/stream", json={"goal": "fixture"}, stream=True
            ) as response:
                assert list(response.iter_lines()) == [
                    b'0:"answer"',
                    b'8:[{"task_id":"task"}]',
                ]
            assert response.raw.closed
        assert len(calls) == 1 and calls[0][1] is None
        assert json.loads(calls[0][2]) == {"goal": "fixture"}


def test_redirects_compressed_and_oversized_responses_close_without_following(
    monkeypatch,
):
    with service() as (origin, calls):
        monkeypatch.setattr(settings, "EXTERNAL_ENGINE_ALLOWED_ORIGINS", [origin])
        with external_engine_session(origin) as session:
            with pytest.raises(ProviderConfigurationError, match="redirects"):
                session.post(
                    origin + "/redirect", headers={"Authorization": "Bearer fixture"}
                )
            with pytest.raises(GenerationExpired, match="Compressed"):
                session.post(origin + "/compressed")
            with session.post(origin + "/large", stream=True) as response:
                with pytest.raises(GenerationExpired, match="response budget"):
                    list(response.iter_lines())
                assert response.raw.closed
        assert [path for path, *_ in calls] == ["/redirect", "/compressed", "/large"]
        with external_engine_session(origin) as session:
            with pytest.raises(GenerationExpired, match="request body"):
                session.post(origin + "/stream", json={"goal": "x" * (2 * 1024 * 1024)})
        assert len(calls) == 3


def test_standalone_timeout_and_active_generation_deadline_bound_real_reads():
    with service() as (origin, calls):
        with ProviderSession(origin, max_seconds=0.04) as session:
            with session.post(origin + "/slow", stream=True) as response:
                started = time.monotonic()
                with pytest.raises((requests.RequestException, GenerationExpired)):
                    list(response.iter_lines(chunk_size=1))
                assert time.monotonic() - started < 1
                assert response.raw.closed
        count = len(calls)

        def expired():
            raise GenerationExpired("Generation deadline exceeded")

        token = current_generation.set(SimpleNamespace(before_outbound=expired))
        try:
            with ProviderSession(origin) as session:
                with pytest.raises(GenerationExpired):
                    session.post(origin + "/stream")
        finally:
            current_generation.reset(token)
        assert len(calls) == count


def test_verification_uses_explicit_token_and_preserves_result(monkeypatch):
    from unittest.mock import Mock
    from contextlib import nullcontext
    from app.rag.chat.chat_flow import ChatFlow

    with service() as (origin, calls):
        monkeypatch.setattr(settings, "EXTERNAL_ENGINE_ALLOWED_ORIGINS", [origin])
        flow = ChatFlow.__new__(ChatFlow)
        flow.engine_config = ChatEngineConfig(
            post_verification_url=origin + "/verify",
            post_verification_token="fixture-token",
        )
        flow._trace_manager = SimpleNamespace(span=lambda **kwargs: nullcontext(Mock()))
        assert (
            flow._post_verification("question", "answer", uuid4(), 1)
            == origin + "/verify/123"
        )
        assert calls[0][1] == "Bearer fixture-token"
        assert "question" in json.loads(calls[0][2])["qa_content"]


def test_early_stream_close_releases_response(monkeypatch):
    with service() as (origin, _):
        monkeypatch.setattr(settings, "EXTERNAL_ENGINE_ALLOWED_ORIGINS", [origin])
        response = None

        def stream():
            nonlocal response
            with external_engine_session(origin) as session:
                with session.post(origin + "/stream", stream=True) as response:
                    yield from response.iter_lines(chunk_size=1)

        iterator = stream()
        assert next(iterator) == b'0:"answer"'
        iterator.close()
        assert response.raw.closed
