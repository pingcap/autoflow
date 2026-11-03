import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from local_embedding_reranker import main as sidecar
from app.rag.local_model_batches import local_model_batches

TOKEN = "fixture-sidecar-token-for-security-tests-0001"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SIDECAR_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("PRE_LOAD_DEFAULT_EMBEDDING_MODEL", "false")
    monkeypatch.setenv("PRE_LOAD_DEFAULT_RERANKER_MODEL", "false")
    sidecar.EMBEDDING_MODEL_DICT.clear()
    sidecar.RERANKER_MODEL_DICT.clear()
    with TestClient(sidecar.app) as client:
        yield client
    sidecar.EMBEDDING_MODEL_DICT.clear()
    sidecar.RERANKER_MODEL_DICT.clear()


def test_model_service_requires_authentication_before_reading_input(
    client, monkeypatch
):
    load = Mock()
    monkeypatch.setattr(sidecar, "get_embedding_model", load)
    assert (
        client.post("/api/v1/embedding", json={"sentences": ["hello"]}).status_code
        == 401
    )
    assert (
        client.post(
            "/api/v1/embedding",
            headers={"Authorization": "Bearer invalid"},
            content=b"x" * (sidecar.MAX_BODY_BYTES + 1),
        ).status_code
        == 401
    )
    load.assert_not_called()
    assert client.get("/api/v1/healthz").status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"sentences": ["hello"], "model": "attacker/custom-python"},
        {"sentences": ["hello"], "model": "/etc/passwd"},
        {"sentences": []},
        {"sentences": ["x"] * 33},
        {"sentences": ["x" * 8193]},
        {"sentences": ["x" * 8192] * 9},
        {"sentences": ["hello"], "trust_remote_code": True},
    ],
)
def test_unapproved_models_and_oversized_work_are_rejected(
    client, monkeypatch, payload
):
    load = Mock()
    monkeypatch.setattr(sidecar, "get_embedding_model", load)
    assert (
        client.post(
            "/api/v1/embedding",
            headers={"Authorization": "Bearer " + TOKEN},
            json=payload,
        ).status_code
        == 422
    )
    load.assert_not_called()


def test_body_limit_and_single_inference_slot(client, monkeypatch):
    headers = {"Authorization": "Bearer " + TOKEN}
    assert (
        client.post(
            "/api/v1/embedding",
            headers=headers,
            content=b"x" * (sidecar.MAX_BODY_BYTES + 1),
        ).status_code
        == 413
    )
    load = Mock()
    monkeypatch.setattr(sidecar, "get_embedding_model", load)
    assert sidecar.INFERENCE_SLOT.acquire(blocking=False)
    try:
        assert (
            client.post(
                "/api/v1/embedding", headers=headers, json={"sentences": ["hello"]}
            ).status_code
            == 429
        )
    finally:
        sidecar.INFERENCE_SLOT.release()
    load.assert_not_called()


def test_approved_models_are_pinned_offline_and_cannot_load_remote_code(
    client, monkeypatch
):
    embedding = Mock()
    embedding.encode.return_value.tolist.return_value = [[1.0, 2.0]]
    reranker = Mock()
    reranker.predict.return_value.tolist.return_value = [0.9]
    embedding_constructor, reranker_constructor = (
        Mock(return_value=embedding),
        Mock(return_value=reranker),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=embedding_constructor, CrossEncoder=reranker_constructor
        ),
    )
    headers = {"Authorization": "Bearer " + TOKEN}
    assert client.post(
        "/api/v1/embedding", headers=headers, json={"sentences": ["hello"]}
    ).json()["embeddings"] == [[1.0, 2.0]]
    assert client.post(
        "/api/v1/reranker", headers=headers, json={"query": "q", "passages": ["p"]}
    ).json()["scores"] == [0.9]
    for constructor in (embedding_constructor, reranker_constructor):
        args = constructor.call_args.kwargs
        assert args["trust_remote_code"] is False
        assert args["local_files_only"] is True
        assert len(args["revision"]) == 40
    assert embedding.encode.call_args.kwargs["batch_size"] == 4
    assert reranker.predict.call_args.kwargs["batch_size"] == 4
    with pytest.raises(HTTPException):
        sidecar.get_embedding_model("attacker/new-model")
    assert embedding_constructor.call_count == 1


def test_large_legitimate_batches_preserve_every_input_in_order():
    texts = [str(i) + "x" * 4096 for i in range(100)]
    batches = list(local_model_batches(texts, query="query"))
    assert sum(batches, []) == texts
    assert all(
        len(batch) <= 32 and sum(map(len, batch)) + 5 <= 65536 for batch in batches
    )
    with pytest.raises(ValueError):
        list(local_model_batches(["x" * 8193]))


def test_model_service_cannot_start_without_a_strong_token(monkeypatch):
    monkeypatch.delenv("SIDECAR_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="SIDECAR_AUTH_TOKEN"):
        with TestClient(sidecar.app):
            pass


def test_backend_batches_send_authentication_and_preserve_embedding_and_ranking(
    client, monkeypatch
):
    import json
    from urllib.parse import urlsplit
    import requests
    from app.rag.embeddings.local.local_embedding import LocalEmbedding
    from app.rag.rerankers.local.local_reranker import LocalRerank
    from llama_index.core.schema import NodeWithScore, TextNode

    calls = []

    def forward(request_session, request, **kwargs):
        assert request.headers["Authorization"] == "Bearer " + TOKEN
        payload = json.loads(request.body)
        calls.append(payload)
        result = client.request(
            request.method,
            urlsplit(request.url).path,
            headers=dict(request.headers),
            content=request.body,
        )
        response = requests.Response()
        response.status_code = result.status_code
        response._content = result.content
        response._content_consumed = True
        response.url = request.url
        return response

    monkeypatch.setattr(requests.Session, "send", forward)
    embed = Mock()
    embed.encode.side_effect = lambda values, **kwargs: SimpleNamespace(
        tolist=lambda: [[float(value)] for value in values]
    )
    ranker = Mock()
    ranker.predict.side_effect = lambda pairs, **kwargs: SimpleNamespace(
        tolist=lambda: [float(pair[1]) for pair in pairs]
    )
    monkeypatch.setattr(sidecar, "get_embedding_model", lambda model_name: embed)
    monkeypatch.setattr(sidecar, "get_reranker_model", lambda model_name: ranker)
    texts = [str(i) for i in range(100)]
    model = LocalEmbedding(
        api_url="http://sidecar/api/v1/embedding", api_key=TOKEN, embed_batch_size=100
    )
    assert model.get_text_embedding_batch(texts) == [[float(i)] for i in range(100)]
    reranker = LocalRerank(
        api_url="http://sidecar/api/v1/reranker", api_key=TOKEN, top_n=3
    )
    ranked = reranker.postprocess_nodes(
        [NodeWithScore(node=TextNode(text=text)) for text in texts], query_str="query"
    )
    assert [node.score for node in ranked] == [99.0, 98.0, 97.0]
    assert len(calls) == 8
