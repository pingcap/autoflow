import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from llama_index.core.base.llms.types import ChatMessage

from app.auth.api_keys import api_key_manager
from app.auth.key_policy import ApiKeyGrant
from app.core.config import settings
from app.models import User, UserSession, ApiKey, Chat
from app.rag.chat.chat_flow import ChatFlow
from app.rag.chat import chat_flow


def chat_request(raw_key, payload):
    body = json.dumps(payload).encode()

    async def receive():
        return {"type": "http.request", "body": body}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": settings.API_V1_STR + "/chats",
            "headers": [(b"authorization", ("Bearer " + raw_key).encode())],
            "scheme": "https",
            "server": ("test", 443),
            "query_string": b"",
        },
        receive,
    )


async def scoped_chat_scenario():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda connection: SQLModel.metadata.create_all(
                connection,
                tables=[User.__table__, UserSession.__table__, ApiKey.__table__],
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE chat_engines (id INTEGER PRIMARY KEY, name TEXT, is_default BOOLEAN, deleted_at DATETIME)"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE chats (id TEXT PRIMARY KEY, user_id TEXT, api_key_id INTEGER, engine_id INTEGER, deleted_at DATETIME)"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO chat_engines VALUES (1, 'permitted', true, NULL), (2, 'forbidden', false, NULL)"
            )
        )
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        user = User(
            email="operator@example.test",
            hashed_password="unused-test-hash",
            is_superuser=True,
            is_verified=True,
        )
        session.add(user)
        await session.commit()
        grant = ApiKeyGrant(scopes=["chat:write"], chat_engine_ids=[1])
        key_a, raw_a = await api_key_manager.acreate_api_key(session, user, "A", grant)
        key_b, raw_b = await api_key_manager.acreate_api_key(session, user, "B", grant)
        owned_chat, human_chat = uuid4(), uuid4()
        await session.execute(
            text("INSERT INTO chats VALUES (:id, :user, :key, 1, NULL)"),
            {"id": owned_chat.hex, "user": user.id.hex, "key": key_a.id},
        )
        await session.execute(
            text("INSERT INTO chats VALUES (:id, :user, NULL, 1, NULL)"),
            {"id": human_chat.hex, "user": user.id.hex},
        )
        await session.commit()
        for payload in [
            {"chat_engine": "permitted"},
            {},
            {"chat_id": str(owned_chat), "chat_engine": "forbidden"},
        ]:
            req = chat_request(raw_a, payload)
            assert (
                await api_key_manager.get_active_user_from_request(session, req) is user
            )
            assert req.state.api_key_engine_name == "permitted"
            assert req.state.api_key_engine_ids == (1,)
        for raw, payload in [
            (raw_a, {"chat_engine": "forbidden"}),
            (raw_a, {"chat_id": str(human_chat)}),
            (raw_b, {"chat_id": str(owned_chat)}),
            (raw_a, {"chat_id": str(uuid4())}),
        ]:
            with pytest.raises(HTTPException) as error:
                await api_key_manager.get_active_user_from_request(
                    session, chat_request(raw, payload)
                )
            assert error.value.status_code == 403
        await session.execute(
            text("UPDATE chat_engines SET deleted_at = CURRENT_TIMESTAMP WHERE id = 1")
        )
        await session.commit()
        with pytest.raises(HTTPException) as error:
            await api_key_manager.get_active_user_from_request(
                session, chat_request(raw_a, {"chat_id": str(owned_chat)})
            )
        assert error.value.status_code == 403
    await engine.dispose()


def test_chat_grants_bind_real_queries_to_engine_and_key_ownership():
    asyncio.run(scoped_chat_scenario())


def test_chat_service_does_not_use_admin_role_to_bypass_key_ownership(monkeypatch):
    owner = SimpleNamespace(id=uuid4(), is_superuser=True)
    chat = Chat(
        id=uuid4(), user_id=owner.id, engine_id=1, api_key_id=100, title="owner chat"
    )
    monkeypatch.setattr(chat_flow.chat_repo, "get", Mock(return_value=chat))
    load = Mock()
    monkeypatch.setattr(chat_flow.ChatEngineConfig, "load_from_db", load)
    with pytest.raises(HTTPException):
        ChatFlow(
            db_session=Mock(),
            user=owner,
            browser_id="guest",
            origin=None,
            chat_messages=[ChatMessage(role="user", content="q")],
            chat_id=chat.id,
            api_key_id=200,
            allowed_engine_ids=[1],
        )
    load.assert_not_called()


def test_changed_engine_scope_is_rechecked_before_creating_chat_or_calling_model(
    monkeypatch,
):
    config = Mock()
    config.get_db_chat_engine.return_value = SimpleNamespace(id=99)
    monkeypatch.setattr(
        chat_flow.ChatEngineConfig, "load_from_db", Mock(return_value=config)
    )
    create = Mock()
    monkeypatch.setattr(chat_flow.chat_repo, "create", create)
    with pytest.raises(HTTPException) as error:
        ChatFlow(
            db_session=Mock(),
            user=SimpleNamespace(id=uuid4(), is_superuser=True),
            browser_id="guest",
            origin=None,
            chat_messages=[ChatMessage(role="user", content="q")],
            api_key_id=200,
            allowed_engine_ids=[1],
        )
    assert error.value.status_code == 404
    create.assert_not_called()
    config.get_llama_llm.assert_not_called()
