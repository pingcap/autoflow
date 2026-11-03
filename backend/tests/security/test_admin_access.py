import inspect
import re
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.main import api_router
from app.api.admin_routes.knowledge_base.graph import routes as graph_routes
from app.auth.users import current_superuser
from app.core.db import get_db_async_session, get_db_session
from app.models import User


@pytest.fixture
def application():
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    app.dependency_overrides[get_db_async_session] = lambda: None
    database = Mock(return_value=object())
    app.dependency_overrides[get_db_session] = lambda: database()
    return app, database


@pytest.mark.parametrize("role,status", [(None, 401), ("member", 403)])
def test_every_admin_route_rejects_unprivileged_access_before_database_work(
    application, role, status
):
    app, database = application
    if role:
        cookie_user = (
            inspect.signature(current_superuser).parameters["user"].default.dependency
        )
        app.dependency_overrides[cookie_user] = lambda: User(
            id=uuid4(),
            email="member@example.test",
            hashed_password="unused",
            is_active=True,
            is_verified=True,
            is_superuser=False,
        )
    routes = {
        path: operations
        for path, operations in app.openapi()["paths"].items()
        if path.startswith("/api/v1/admin/")
    }
    assert len(routes) > 40
    with TestClient(app) as client:
        for pattern, operations in routes.items():
            path = re.sub(r"\{[^}]+\}", "1", pattern)
            for method in operations.keys() & {"get", "post", "put", "patch", "delete"}:
                response = client.request(method, path)
                assert response.status_code == status, (
                    method,
                    path,
                    response.status_code,
                )
    database.assert_not_called()


def test_authorized_graph_stream_retains_role_check_and_has_no_wildcard_cors(
    application, monkeypatch
):
    app, database = application
    cookie_user = (
        inspect.signature(current_superuser).parameters["user"].default.dependency
    )
    app.dependency_overrides[cookie_user] = lambda: User(
        id=uuid4(),
        email="admin@example.test",
        hashed_password="unused",
        is_active=True,
        is_verified=True,
        is_superuser=True,
    )
    monkeypatch.setattr(
        graph_routes.knowledge_base_repo, "must_get", lambda *args: object()
    )
    store = Mock()
    store.stream_entire_knowledge_graph.return_value = [
        {"entities": [], "relationships": []}
    ]
    monkeypatch.setattr(graph_routes, "get_kb_tidb_graph_store", lambda *args: store)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/admin/knowledge_bases/1/graph/entire_graph/stream",
            headers={"Origin": "https://attacker.example"},
        )
    assert response.status_code == 200
    assert '"type": "complete"' in response.text
    assert "access-control-allow-origin" not in response.headers
    assert response.headers["cache-control"] == "no-store"
    database.assert_called_once()
