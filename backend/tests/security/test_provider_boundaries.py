import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings
from app.models import LLM, LLMUpdate
from app.rag.llms.provider import LLMProvider
from app.rag.llms.resolver import resolve_llm
from app.repositories.llm import llm_repo
from app.security.provider_config import (
    ProviderConfigurationError,
    prepare_provider_config,
)
from app.security.provider_http import ProviderSession, openai_http_clients


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1",
        "http://169.254.169.254/latest/meta-data",
        "http://2130706433",
        "http://[::1]",
        "https://api.openai.com.evil.test",
        "https://api.openai.com@evil.test",
        "https://api.openai.com\\@evil.test",
        "https://api.openai.com:444/v1",
        "https://api.openai.com/v1?key=secret",
        "https://api.openai.com./v1",
    ],
)
def test_unregistered_destinations_cannot_create_a_model_client(url):
    with pytest.raises(ProviderConfigurationError):
        resolve_llm(LLMProvider.OPENAI, "gpt-4o", {"api_base": url}, "fixture-key")


@pytest.mark.parametrize(
    "config",
    [
        {"http_options": {"base_url": "http://127.0.0.1"}},
        {"client_kwargs": {"verify": False}},
        {"default_headers": {"Host": "internal"}},
        {"additional_kwargs": {"extra_headers": {"Host": "internal"}}},
        {"additional_kwargs": {"api_base": "http://localhost"}},
        {"max_tokens": 999999},
        {"timeout": 999999},
        {"temperature": float("nan")},
    ],
)
def test_sdk_transport_overrides_and_unbounded_parameters_are_rejected(config):
    with pytest.raises(ProviderConfigurationError):
        prepare_provider_config("llm", "openai", "gpt-4o", config, "fixture-key")


def test_empty_model_credentials_never_inherit_server_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-server-secret")
    with pytest.raises(
        ProviderConfigurationError, match="Explicit provider credentials"
    ):
        resolve_llm(LLMProvider.OPENAI, "gpt-4o", {}, None)
    with pytest.raises(ProviderConfigurationError):
        prepare_provider_config(
            "llm",
            "vertex_ai",
            "gemini-2.5-flash",
            {},
            {
                "type": "service_account",
                "token_uri": "http://169.254.169.254",
                "private_key": "fixture",
            },
        )
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    with pytest.raises(ProviderConfigurationError, match="ambient"):
        prepare_provider_config(
            "llm", "gemini", "models/gemini-2.5-flash", {}, "fixture-key"
        )


def test_registered_private_endpoint_requires_operator_configuration(monkeypatch):
    config = {"base_url": "http://ollama:11434"}
    with pytest.raises(ProviderConfigurationError):
        prepare_provider_config("llm", "ollama", "llama3.2", config, None)
    monkeypatch.setattr(settings, "PROVIDER_ALLOWED_ORIGINS", ["http://ollama:11434"])
    assert (
        prepare_provider_config("llm", "ollama", "llama3.2", config, None)["base_url"]
        == config["base_url"]
    )


@pytest.mark.parametrize("credential", [None, "fixture-explicit-key"])
def test_real_ollama_and_dspy_never_substitute_ambient_keys(monkeypatch, credential):
    import json
    import socket
    import litellm
    from llama_index.core.base.llms.types import ChatMessage
    from app.rag.llms.dspy import get_dspy_lm_by_llama_llm

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append((self.path, self.headers.get("Authorization")))
            response = json.dumps(
                {
                    "model": "llama3.2",
                    "created_at": "2026-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "fixture answer"},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 1,
                    "eval_count": 2,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    connect = socket.socket.connect

    def local_only(sock, address):
        assert isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
    for proxy in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        monkeypatch.delenv(proxy, raising=False)
    monkeypatch.setattr(settings, "PROVIDER_ALLOWED_ORIGINS", [origin])
    monkeypatch.setenv("OLLAMA_API_KEY", "forbidden-ambient-key")
    monkeypatch.setattr(litellm, "ollama_key", "forbidden-global-ollama-key")
    monkeypatch.setattr(litellm, "api_key", "forbidden-global-key")
    llm = resolve_llm(
        LLMProvider.OLLAMA,
        "llama3.2",
        {"base_url": origin, "request_timeout": 2},
        credential,
    )
    dspy_lm = None
    try:
        assert (
            llm.chat([ChatMessage(role="user", content="fixture")]).message.content
            == "fixture answer"
        )
        dspy_lm = get_dspy_lm_by_llama_llm(llm)
        assert dspy_lm(prompt="fixture") == ["fixture answer"], [
            path for path, _ in calls
        ]
        expected = "Bearer " + (credential or "autoflow-local-no-auth")
        assert calls == [("/api/chat", expected), ("/api/chat", expected)]
    finally:
        if dspy_lm is not None:
            dspy_lm.close()
        llm.client._client.close()
        asyncio.run(llm.async_client._client.aclose())
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_config_updates_cannot_reuse_stored_credentials_and_refresh_stale_rows(
    monkeypatch,
):
    monkeypatch.setattr(
        settings, "PROVIDER_ALLOWED_ORIGINS", ["https://gateway.example.test"]
    )
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine, tables=[LLM.__table__])
    with Session(engine) as session:
        llm = llm_repo.create(
            session,
            LLM(
                name="fixture",
                model="gpt-4o",
                provider=LLMProvider.OPENAI,
                config={},
                credentials="old-fixture-key",
            ),
        )
        model_id = llm.id
    with Session(engine) as stale, Session(engine) as current:
        stale_llm = stale.get(LLM, model_id)
        current_llm = current.get(LLM, model_id)
        target = {"api_base": "https://gateway.example.test/v1"}
        with pytest.raises(HTTPException, match="fresh credentials"):
            llm_repo.update(current, current_llm, LLMUpdate(config=target))
        current.rollback()
        llm_repo.update(
            current,
            current_llm,
            LLMUpdate(config=target, credentials="new-fixture-key"),
        )
        # A stale editor cannot return the new secret to the old endpoint.
        with pytest.raises(HTTPException, match="fresh credentials"):
            llm_repo.update(
                stale,
                stale_llm,
                LLMUpdate(
                    config={"api_base": "https://api.openai.com/v1", "temperature": 0.2}
                ),
            )
        stale.rollback()
        current.refresh(current_llm)
        assert current_llm.config["api_base"] == target["api_base"]
        assert current_llm.credentials == "new-fixture-key"
        updated = llm_repo.update(
            current, current_llm, LLMUpdate(config={**target, "temperature": 0.2})
        )
        assert updated.credentials == "new-fixture-key"
    engine.dispose()


@contextmanager
def redirect_server():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/secret-target")
            self.end_headers()

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
        thread.join()


def test_sync_and_async_model_transports_do_not_follow_redirects():
    with redirect_server() as (origin, calls):
        with ProviderSession(origin) as session:
            with pytest.raises(ProviderConfigurationError, match="redirects"):
                session.get(origin + "/start")
            with pytest.raises(ProviderConfigurationError, match="destination"):
                session.get("http://169.254.169.254")
        clients = openai_http_clients()
        with clients["http_client"] as client:
            assert client.get(origin + "/sync").status_code == 302

        async def request():
            async with clients["async_http_client"] as client:
                assert (await client.get(origin + "/async")).status_code == 302

        asyncio.run(request())
        assert calls == ["/start", "/sync", "/async"]


def test_provider_errors_do_not_return_or_log_credentials(monkeypatch, caplog):
    from app.api.admin_routes.llm import routes

    secret = "fixture-private-provider-secret"
    monkeypatch.setattr(
        routes,
        "resolve_llm",
        Mock(side_effect=RuntimeError("remote response contained " + secret)),
    )
    result = routes.test_llm(
        SimpleNamespace(
            provider=LLMProvider.OPENAI, model="gpt-4o", config={}, credentials=secret
        ),
        Mock(),
    )
    assert not result.success
    assert secret not in result.error + caplog.text
    assert result.stage == "configuration"
    assert result.error_type == "RuntimeError"


@pytest.mark.parametrize("stage", ["llm", "dspy"])
def test_model_diagnostics_distinguish_provider_and_adapter_without_error_bodies(
    monkeypatch, stage, caplog
):
    from app.api.admin_routes.llm import routes
    from app.rag.llms import dspy as conversion

    secret = "fixture-private-upstream-body"
    llm = Mock()
    monkeypatch.setattr(routes, "resolve_llm", Mock(return_value=llm))
    if stage == "llm":
        llm.chat.side_effect = RuntimeError(secret)
    else:
        monkeypatch.setattr(
            conversion,
            "get_dspy_lm_by_llama_llm",
            Mock(side_effect=RuntimeError(secret)),
        )
    result = routes.test_llm(
        SimpleNamespace(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            config={},
            credentials="fixture-key",
        ),
        Mock(),
    )
    assert result.stage == stage
    assert result.error_type == "RuntimeError"
    assert not result.success
    assert secret not in result.model_dump_json() + caplog.text


def test_openai_model_verification_runs_both_sdks_against_only_loopback(monkeypatch):
    import json
    import socket
    from app.api.admin_routes.llm import routes

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("Authorization")))
            content = (
                json.dumps({"answer": 2})
                if payload.get("response_format")
                else "[[ ## answer ## ]]\n2\n[[ ## completed ## ]]"
            )
            body = json.dumps(
                {
                    "id": "chatcmpl-fixture",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    connect = socket.socket.connect

    def local_only(sock, address):
        assert isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}, (
            "Test attempted non-loopback access"
        )
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
    for proxy in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        monkeypatch.delenv(proxy, raising=False)
    monkeypatch.setattr(settings, "PROVIDER_ALLOWED_ORIGINS", [origin])
    try:
        result = routes.test_llm(
            SimpleNamespace(
                provider=LLMProvider.OPENAI,
                model="gpt-4o-mini",
                config={"api_base": origin + "/v1", "timeout": 2},
                credentials="test-only-provider-key",
            ),
            Mock(),
        )
        assert result.success, (result.stage, result.error_type)
        assert calls == [
            ("/v1/chat/completions", "Bearer test-only-provider-key"),
            ("/v1/chat/completions", "Bearer test-only-provider-key"),
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_supported_provider_clients_still_construct_with_explicit_credentials(
    monkeypatch,
):
    from google.genai.models import Models
    from google.genai.types import Model
    from app.rag.embeddings.resolver import resolve_embed_model
    from app.rag.embeddings.provider import EmbeddingProvider
    from app.rag.rerankers.resolver import resolve_reranker
    from app.rag.rerankers.provider import RerankerProvider

    monkeypatch.setattr(
        settings,
        "PROVIDER_ALLOWED_ORIGINS",
        [
            "http://localhost:11434",
            "http://local-embedding-reranker:5001",
            "https://example.openai.azure.com",
            "http://localhost:9997",
            "http://localhost:8000",
        ],
    )
    monkeypatch.setattr(
        Models,
        "get",
        Mock(
            return_value=Model(
                name="models/gemini-2.5-flash",
                input_token_limit=1000000,
                output_token_limit=8192,
            )
        ),
    )
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    llms = [
        (LLMProvider.OPENAI, "gpt-4o", {}),
        (LLMProvider.OPENAI_LIKE, "custom-model", {}),
        (LLMProvider.GEMINI, "models/gemini-2.5-flash", {}),
        (LLMProvider.OLLAMA, "llama3.2", {}),
        (
            LLMProvider.AZURE_OPENAI,
            "gpt-4o",
            {
                "azure_endpoint": "https://example.openai.azure.com",
                "api_version": "2024-02-15-preview",
                "engine": "deployment",
            },
        ),
    ]
    for provider, model, config in llms:
        resolved = resolve_llm(provider, model, config, "fixture-only")
        assert resolved is not None
    for provider, model in [
        (EmbeddingProvider.OPENAI, "text-embedding-3-small"),
        (EmbeddingProvider.OPENAI_LIKE, "custom-model"),
        (EmbeddingProvider.JINA, "jina-embeddings-v3"),
        (EmbeddingProvider.COHERE, "embed-english-v3.0"),
        (EmbeddingProvider.LOCAL, "BAAI/bge-m3"),
        (EmbeddingProvider.OLLAMA, "nomic-embed-text"),
    ]:
        assert (
            resolve_embed_model(
                provider, model, {}, "fixture-only-sidecar-key-0000000000000000"
            )
            is not None
        )
    for provider, model in [
        (RerankerProvider.JINA, "jina-reranker-v2-base-multilingual"),
        (RerankerProvider.COHERE, "rerank-english-v3.0"),
        (RerankerProvider.LOCAL, "BAAI/bge-reranker-v2-m3"),
        (RerankerProvider.XINFERENCE, "bge-reranker-v2-m3"),
        (RerankerProvider.VLLM, "BAAI/bge-reranker-v2-m3"),
    ]:
        assert (
            resolve_reranker(
                provider, model, 2, {}, "fixture-only-sidecar-key-0000000000000000"
            )
            is not None
        )
