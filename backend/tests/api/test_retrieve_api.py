from fastapi.testclient import TestClient
from app.api_server import app

client = TestClient(app)


def test_admin_retrieve_documents(test_api_key):
    response = client.get(
        "/api/v1/admin/retrieve/documents",
        headers={"Authorization": f"Bearer {test_api_key}"},
        params={
            "chat_engine": "default",
            "question": "what is tidb",
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    assert response.json() is not None


def test_admin_embedding_retrieve(test_api_key):
    response = client.get(
        "/api/v1/admin/embedding_retrieve",
        headers={"Authorization": f"Bearer {test_api_key}"},
        params={
            "chat_engine": "default",
            "question": "what is tidb",
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    assert response.json() is not None
