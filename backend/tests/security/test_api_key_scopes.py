import asyncio
import json
from datetime import datetime, timedelta, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from pydantic import ValidationError

from app.auth.api_keys import ApiKeyManager, hash_api_key
from app.auth.key_policy import ApiKeyGrant
from app.core.config import settings


def request(path, ids=None, method="POST"):
    body = json.dumps({"retrieval_config": {"knowledge_base_ids": ids}}).encode()

    async def receive():
        return {"type": "http.request", "body": body}

    return Request(
        {
            "type": "http",
            "method": method,
            "path": settings.API_V1_STR + path,
            "headers": [(b"authorization", b"Bearer ta-" + b"A" * 50)],
            "scheme": "https",
            "server": ("test", 443),
            "query_string": b"",
        },
        receive,
    )


@pytest.fixture
def authenticated(monkeypatch):
    manager = ApiKeyManager()
    key = SimpleNamespace(id=10, scopes=["retrieve:chunks"], knowledge_base_ids=[7, 8])
    user = SimpleNamespace(id=uuid4(), is_superuser=True)
    monkeypatch.setattr(manager, "_authenticate", AsyncMock(return_value=(key, user)))
    session = SimpleNamespace(exec=AsyncMock(return_value=Mock(all=lambda: [7])))
    return manager, key, user, session


@pytest.mark.parametrize(
    "path,method,ids",
    [
        ("/api-keys", "GET", [7]),
        ("/api-keys", "POST", [7]),
        ("/admin/site-settings", "GET", [7]),
        ("/documents/7/download", "GET", [7]),
        ("/chats", "POST", [7]),
        ("/retrieve/chunks", "GET", [7]),
        ("/retrieve/knowledge_graph", "POST", [7]),
        ("/retrieve/chunks", "POST", [99]),
        ("/retrieve/chunks", "POST", [7, 99]),
        ("/retrieve/chunks", "POST", []),
        ("/retrieve/chunks", "POST", None),
        ("/retrieve/chunks", "POST", [True]),
        ("/retrieve/chunks", "POST", ["7"]),
    ],
)
def test_key_never_inherits_owners_admin_or_unrelated_data_access(
    authenticated, path, method, ids
):
    manager, _, _, session = authenticated
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            manager.get_active_user_from_request(session, request(path, ids, method))
        )
    assert error.value.status_code == 403
    session.exec.assert_not_called()


def test_valid_scoped_retrieval_authenticates_and_records_key_identity(authenticated):
    manager, _, user, session = authenticated
    req = request("/retrieve/chunks", [7])
    assert asyncio.run(manager.get_active_user_from_request(session, req)) is user
    assert req.state.api_key_id == 10
    assert req.state.user_id == str(user.id)
    statement = session.exec.call_args.args[0]
    assert "deleted_at IS NULL" in str(statement)


def test_deleted_knowledge_base_is_no_longer_available_to_key(authenticated):
    manager, _, _, session = authenticated
    session.exec.return_value = Mock(all=lambda: [])
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            manager.get_active_user_from_request(
                session, request("/retrieve/chunks", [7])
            )
        )
    assert error.value.status_code == 403


@pytest.mark.parametrize(
    "overrides",
    [
        {"scopes": ["admin"]},
        {"knowledge_base_ids": []},
        {"knowledge_base_ids": [True]},
        {"knowledge_base_ids": [-1]},
        {"expires_in_days": 0},
        {"expires_in_days": 91},
    ],
)
def test_key_creation_cannot_request_wildcard_scope_or_unbounded_lifetime(overrides):
    values = {
        "scopes": ["retrieve:chunks"],
        "knowledge_base_ids": [7],
        "expires_in_days": 30,
    }
    with pytest.raises(ValidationError):
        ApiKeyGrant(**(values | overrides))


@pytest.mark.parametrize("expiry", [None, datetime.now(UTC) - timedelta(seconds=1)])
def test_legacy_or_expired_key_is_rejected(expiry):
    key = SimpleNamespace(expires_at=expiry)
    session = SimpleNamespace(
        exec=AsyncMock(return_value=Mock(first=lambda: key)), get=AsyncMock()
    )
    assert (
        asyncio.run(
            ApiKeyManager().get_active_user_by_raw_api_key(session, "ta-" + "A" * 50)
        )
        is None
    )
    session.get.assert_not_called()


def test_new_key_hash_cannot_match_the_old_unrestricted_worker_format():
    digest = hash_api_key("ta-" + "A" * 50)
    assert digest.startswith("sha256-v2$")
    assert len(digest.removeprefix("sha256-v2$")) == 64
    assert "ta-" not in digest
