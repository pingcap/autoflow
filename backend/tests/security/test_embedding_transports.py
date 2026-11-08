"""Embedding HTTP budgets and identity, with fake credentials and owned sockets."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
from threading import Event, Thread
import time
from types import SimpleNamespace

import pytest
from ollama import ResponseError

from app.core.config import settings
from app.rag.embeddings.resolver import resolve_embed_model
from app.security import bedrock_http
from app.security.generation import current_generation
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError
from tests.security.test_bedrock_transport import scope


@pytest.fixture
def endpoint(monkeypatch, tmp_path):
    state = SimpleNamespace(calls=[], mode="normal", started=Event(), closed=Event())

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append((self.path, dict(self.headers), body))
            texts = body.get("texts", [body.get("inputText", body.get("input", ""))])
            vectors = [[float(index), 0.2, 0.3] for index, _ in enumerate(texts)]
            payload = {
                "embedding": vectors[0],
                "embeddings": vectors,
                "model": "fixture",
            }
            if state.mode in {"oversized", "slow"}:
                payload["padding"] = "x" * (4 * 1024 * 1024 + 1)
            data = json.dumps(payload).encode()
            self.send_response(302 if state.mode == "redirect" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if state.mode == "compressed":
                self.send_header("Content-Encoding", "gzip")
            if state.mode == "redirect":
                self.send_header("Location", state.origin + "/unexpected")
            self.end_headers()
            state.started.set()
            try:
                if state.mode == "slow":
                    for index in range(0, len(data), 256):
                        self.wfile.write(data[index : index + 256])
                        self.wfile.flush()
                        time.sleep(0.01)
                else:
                    self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                state.closed.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.origin = f"http://127.0.0.1:{server.server_port}"
    connect, dns = socket.socket.connect, socket.getaddrinfo

    def owned_connect(sock, address):
        assert address == ("127.0.0.1", server.server_port)
        return connect(sock, address)

    def owned_dns(host, port, *args, **kwargs):
        assert host == "127.0.0.1" and port == server.server_port
        return dns(host, port, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", owned_connect)
    monkeypatch.setattr(socket, "getaddrinfo", owned_dns)
    monkeypatch.setattr(bedrock_http, "bedrock_endpoint", lambda region: state.origin)
    monkeypatch.setattr(settings, "PROVIDER_ALLOWED_ORIGINS", [state.origin])
    for key, value in {
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": "http://127.0.0.1:1",
        "AWS_SESSION_TOKEN": "fixture-ambient-token",
        "AWS_ACCESS_KEY_ID": "fixture-ambient-id",
        "AWS_SECRET_ACCESS_KEY": "fixture-ambient-key",
        "AWS_PROFILE": "missing-profile",
        "AWS_CONFIG_FILE": str(tmp_path / "bad-config"),
        "AWS_SHARED_CREDENTIALS_FILE": str(tmp_path / "bad-credentials"),
        "AWS_CA_BUNDLE": str(tmp_path / "missing-ca"),
        "OLLAMA_API_KEY": "fixture-ambient-ollama-key",
        "OLLAMA_HOST": "http://127.0.0.1:1",
        "HTTP_PROXY": "http://127.0.0.1:1",
        "HTTPS_PROXY": "http://127.0.0.1:1",
        "ALL_PROXY": "http://127.0.0.1:1",
    }.items():
        monkeypatch.setenv(key, value)
    (tmp_path / "bad-config").write_text("invalid AWS profile")
    (tmp_path / "bad-credentials").write_text("invalid AWS credentials")
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def embedding(provider, endpoint, model=None, config=None):
    if provider == "bedrock":
        return resolve_embed_model(
            provider,
            model or "amazon.titan-embed-text-v2:0",
            config or {},
            {
                "aws_access_key_id": "fixture-explicit-id",
                "aws_secret_access_key": "fixture-explicit-secret",
                "aws_region_name": "us-west-2",
            },
        )
    return resolve_embed_model(
        provider,
        "fixture-embedding",
        {"base_url": endpoint.origin + "/gateway", **(config or {})},
        "fixture-explicit-key",
    )


def invoke(model, asynchronous, text="fixture query"):
    return (
        asyncio.run(model.aget_query_embedding(text))
        if asynchronous
        else model.get_query_embedding(text)
    )


@pytest.mark.parametrize("provider", ["bedrock", "ollama"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_embedding_identity_copies_and_attempt_accounting(
    endpoint, provider, asynchronous
):
    model = embedding(provider, endpoint)
    for candidate in (model, model.model_copy(), model):
        generation = scope()
        token = current_generation.set(generation)
        try:
            assert invoke(candidate, asynchronous) == [0.0, 0.2, 0.3]
            assert generation.provider_calls == 1
        finally:
            current_generation.reset(token)
    assert len(endpoint.calls) == 3
    for path, headers, body in endpoint.calls:
        normalized = {key.lower(): value for key, value in headers.items()}
        assert "x-amz-security-token" not in normalized
        if provider == "bedrock":
            assert path == "/model/amazon.titan-embed-text-v2%3A0/invoke"
            assert "Credential=fixture-explicit-id/" in normalized["authorization"]
            assert body == {"inputText": "fixture query"}
        else:
            assert path == "/gateway/api/embed"
            assert normalized["authorization"] == "Bearer fixture-explicit-key"
            assert body["model"] == "fixture-embedding"
            assert body["input"] == "fixture query"


@pytest.mark.parametrize("provider", ["bedrock", "ollama"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["oversized", "compressed", "redirect"])
def test_embedding_rejects_response_escape_paths(
    endpoint, provider, asynchronous, mode
):
    endpoint.mode = mode
    model = embedding(provider, endpoint)
    token = current_generation.set(scope())
    try:
        expected = GenerationExpired
        if mode == "redirect":
            expected = (
                ProviderConfigurationError if provider == "bedrock" else ResponseError
            )
        with pytest.raises(expected):
            invoke(model, asynchronous)
    finally:
        current_generation.reset(token)
    assert len(endpoint.calls) == 1


@pytest.mark.parametrize("provider", ["bedrock", "ollama"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["attempts", "expired", "body"])
def test_embedding_refuses_unadmitted_or_oversized_request(
    endpoint, provider, asynchronous, mode
):
    generation = scope(
        seconds=-1 if mode == "expired" else 10, attempts=0 if mode == "attempts" else 1
    )
    model = embedding(provider, endpoint)
    token = current_generation.set(generation)
    try:
        with pytest.raises(GenerationExpired):
            invoke(
                model,
                asynchronous,
                "x" * (2 * 1024 * 1024 + 1) if mode == "body" else "fixture query",
            )
    finally:
        current_generation.reset(token)
    assert endpoint.calls == []


@pytest.mark.parametrize("provider", ["bedrock", "ollama"])
def test_async_embedding_cancellation_closes_owned_connection(endpoint, provider):
    endpoint.mode = "slow"
    model = embedding(provider, endpoint)

    async def run():
        token = current_generation.set(scope())
        try:
            task = asyncio.create_task(model.aget_query_embedding("fixture query"))
            assert await asyncio.to_thread(endpoint.started.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await asyncio.to_thread(endpoint.closed.wait, 3)
        finally:
            current_generation.reset(token)

    asyncio.run(run())


def test_cohere_embedding_batch_retains_input_type_and_order(endpoint):
    model = embedding("bedrock", endpoint, model="cohere.embed-multilingual-v3")
    assert model.get_text_embedding_batch(["first", "second"]) == [
        [0.0, 0.2, 0.3],
        [1.0, 0.2, 0.3],
    ]
    assert endpoint.calls[0][2] == {
        "texts": ["first", "second"],
        "input_type": "search_document",
    }
    assert asyncio.run(model.aget_query_embedding("question")) == [0.0, 0.2, 0.3]
    assert endpoint.calls[1][2] == {"texts": ["question"], "input_type": "search_query"}


@pytest.mark.parametrize("provider", ["bedrock", "ollama"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_embedding_deadline_interrupts_slow_response(endpoint, provider, asynchronous):
    endpoint.mode = "slow"
    model = embedding(provider, endpoint)
    token = current_generation.set(scope(seconds=0.3))
    try:
        with pytest.raises(GenerationExpired):
            invoke(model, asynchronous)
        assert endpoint.closed.wait(3)
    finally:
        current_generation.reset(token)
    assert len(endpoint.calls) == 1


@pytest.mark.parametrize("provider", ["bedrock", "ollama"])
def test_embedding_response_is_bounded_without_chat_admission(endpoint, provider):
    endpoint.mode = "oversized"
    model = embedding(provider, endpoint)
    assert current_generation.get() is None
    with pytest.raises(GenerationExpired):
        model.get_query_embedding("fixture standalone")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_titan_embedding_options_are_preserved(endpoint, asynchronous):
    model = embedding(
        "bedrock",
        endpoint,
        config={"additional_kwargs": {"dimensions": 256, "normalize": True}},
    )
    assert invoke(model, asynchronous) == [0.0, 0.2, 0.3]
    assert endpoint.calls[0][2] == {
        "inputText": "fixture query",
        "dimensions": 256,
        "normalize": True,
    }


def test_ollama_embedding_instructions_and_batches_are_preserved(endpoint):
    model = embedding(
        "ollama",
        endpoint,
        config={
            "query_instruction": "query:",
            "text_instruction": "document:",
            "ollama_additional_kwargs": {"num_ctx": 4096},
        },
    )
    assert (
        model.get_text_embedding_batch([" first ", "second"]) == [[0.0, 0.2, 0.3]] * 2
    )
    assert asyncio.run(model.aget_query_embedding(" question ")) == [0.0, 0.2, 0.3]
    assert [call[2]["input"] for call in endpoint.calls] == [
        "document: first",
        "document: second",
        "query: question",
    ]
    assert all(call[2]["options"] == {"num_ctx": 4096} for call in endpoint.calls)
