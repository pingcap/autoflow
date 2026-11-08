"""Existing embedding settings retain their wire behavior and explicit identity."""

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from app.core.config import settings
from app.rag.embeddings.provider import EmbeddingProvider
from app.rag.embeddings.resolver import resolve_embed_model
from app.security.provider_config import (
    ProviderConfigurationError,
    prepare_provider_config,
    validate_provider_write,
)


@contextmanager
def embedding_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), body))
            payload = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": [0.1, 0.2, 0.3],
                        }
                    ],
                    "model": body["model"],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize(
    "provider", [EmbeddingProvider.OPENAI_LIKE, EmbeddingProvider.GITEEAI]
)
def test_openai_compatible_embedding_dimensions_reach_sync_and_async_requests(
    monkeypatch, provider
):
    with embedding_server() as (origin, requests):
        monkeypatch.setattr(settings, "PROVIDER_ALLOWED_ORIGINS", [origin])
        model = resolve_embed_model(
            provider,
            "embedding-fixture",
            {"api_base": origin + "/v1", "dimensions": 3},
            "fixture-model-key",
        )
        try:
            assert model.get_text_embedding("fixture text") == [0.1, 0.2, 0.3]
            assert asyncio.run(model.aget_query_embedding("fixture query")) == [
                0.1,
                0.2,
                0.3,
            ]
            assert len(requests) == 2
            for path, headers, body in requests:
                assert path == "/v1/embeddings"
                assert headers["Authorization"] == "Bearer fixture-model-key"
                assert body["dimensions"] == 3
                assert body["model"] == "embedding-fixture"
        finally:
            model._client.close()
            asyncio.run(model._aclient.close())


@pytest.mark.parametrize("deployment_field", ["azure_deployment", "deployment_name"])
def test_azure_embedding_deployment_and_explicit_api_key_survive_ambient_identity(
    monkeypatch, deployment_field
):
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "fixture-ambient-token")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fixture-ambient-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://ambient.example.test")
    with embedding_server() as (origin, requests):
        monkeypatch.setattr(settings, "PROVIDER_ALLOWED_ORIGINS", [origin])
        model = resolve_embed_model(
            EmbeddingProvider.AZURE_OPENAI,
            "text-embedding-3-small",
            {
                "azure_endpoint": origin + "/gateway",
                deployment_field: "owned-embedding",
                "api_version": "2024-02-01",
                "dimensions": 3,
            },
            "fixture-azure-key",
        )
        try:
            assert model.get_text_embedding("fixture text") == [0.1, 0.2, 0.3]
            assert asyncio.run(model.aget_query_embedding("fixture query")) == [
                0.1,
                0.2,
                0.3,
            ]
            copied = model._get_client().with_options()
            copied.embeddings.create(
                model="text-embedding-3-small", input="fixture copy", dimensions=3
            )
            assert len(requests) == 3
            for path, headers, body in requests:
                assert (
                    path
                    == "/gateway/openai/deployments/owned-embedding/embeddings?api-version=2024-02-01"
                )
                normalized = {key.lower(): value for key, value in headers.items()}
                assert normalized["api-key"] == "fixture-azure-key"
                assert "authorization" not in normalized
                assert body["dimensions"] == 3
        finally:
            if model._client:
                model._client.close()
            if model._aclient:
                asyncio.run(model._aclient.close())
            model._http_client.close()
            asyncio.run(model._async_http_client.aclose())


def test_azure_embedding_deployment_update_requires_new_credentials(monkeypatch):
    monkeypatch.setattr(
        settings, "PROVIDER_ALLOWED_ORIGINS", ["https://azure.example.test"]
    )
    config = {
        "azure_endpoint": "https://azure.example.test",
        "azure_deployment": "original",
        "api_version": "2024-02-01",
    }
    item = SimpleNamespace(
        provider="azure_openai",
        model="text-embedding-3-small",
        config=config,
        credentials="fixture-old-key",
    )
    changed = dict(config, azure_deployment="replacement")
    with pytest.raises(HTTPException, match="fresh credentials"):
        validate_provider_write("embedding", item, {"config": changed})
    accepted = validate_provider_write(
        "embedding", item, {"config": changed, "credentials": "fixture-new-key"}
    )
    assert accepted["azure_deployment"] == "replacement"
    assert item.config == config


@pytest.mark.parametrize("provider", ["vertex_ai", "anthropic_vertex"])
def test_versioned_vertex_model_identifier_is_preserved(provider):
    model = "claude-3-5-sonnet-v2@20241022"
    credentials = {
        "type": "service_account",
        "token_uri": "https://oauth2.googleapis.com/token",
        "project_id": "fixture-project",
    }
    assert (
        prepare_provider_config("llm", provider, model, {}, credentials)["max_tokens"]
        == 4096
    )
    for hostile in [
        "https://host/model@version",
        "model@//host",
        "../model@1",
        "model@1?key=value",
        "model@1@2",
    ]:
        with pytest.raises(ProviderConfigurationError):
            prepare_provider_config("llm", provider, hostile, {}, credentials)
