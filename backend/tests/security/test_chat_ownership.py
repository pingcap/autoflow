from datetime import datetime, UTC
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from llama_index.core.base.llms.types import ChatMessage
from pydantic import ValidationError

from app.auth.browser_session import create_browser_session, verify_browser_session
from app.auth.chat_access import user_can_edit_chat, user_can_view_chat
from app.api.routes.chat import ChatRequest
from app.models import Chat, ChatVisibility
from app.rag.chat.chat_flow import ChatFlow
from app.rag.chat.config import ChatEngineConfig


SECRET = "test-only-browser-secret-with-sufficient-entropy"


def chat(**overrides):
    return Chat(
        title="private question",
        user_id=None,
        browser_id="alice",
        visibility=ChatVisibility.PRIVATE,
        **overrides,
    )


def test_guest_ownership_requires_authenticated_browser_identity():
    private = chat()
    assert user_can_view_chat(private, None, "alice")
    assert user_can_edit_chat(private, None, "alice")
    for browser_id in [None, "", "bob"]:
        assert not user_can_view_chat(private, None, browser_id)
        assert not user_can_edit_chat(private, None, browser_id)
    private.browser_id = None
    assert not user_can_edit_chat(private, None, None)


def test_public_sharing_grants_read_access_but_never_write_access():
    public = chat()
    public.visibility = ChatVisibility.PUBLIC
    assert user_can_view_chat(public, None, "bob")
    assert not user_can_edit_chat(public, None, "bob")
    public.deleted_at = datetime.now(UTC)
    assert not user_can_view_chat(public, None, "alice")


def test_a_browser_id_never_grants_access_to_another_accounts_chat():
    private = chat()
    private.user_id = uuid4()
    owner = SimpleNamespace(id=private.user_id, is_superuser=False)
    other = SimpleNamespace(id=uuid4(), is_superuser=False)
    assert user_can_edit_chat(private, owner, "other-browser")
    assert not user_can_edit_chat(private, other, "alice")
    assert not user_can_view_chat(private, None, "alice")


def test_browser_cookie_is_authenticated_and_expires():
    browser_id, cookie = create_browser_session(SECRET, now=1000)
    assert verify_browser_session(cookie, SECRET, 300, now=1100) == browser_id
    for invalid in [
        browser_id,
        cookie.replace(browser_id, "f" * 32),
        cookie + "x",
        "x" * 1000,
        None,
    ]:
        assert verify_browser_session(invalid, SECRET, 300, now=1100) is None
    assert verify_browser_session(cookie, "another-secret", 300, now=1100) is None
    assert verify_browser_session(cookie, SECRET, 300, now=1300) is None
    assert verify_browser_session(cookie, SECRET, 300, now=900) is None


def test_foreign_chat_continuation_fails_before_history_or_model_access(monkeypatch):
    from app.rag.chat import chat_flow

    private = chat()
    monkeypatch.setattr(chat_flow.chat_repo, "get", lambda *args: private)
    history = Mock()
    config = Mock()
    monkeypatch.setattr(chat_flow.chat_repo, "get_messages", history)
    monkeypatch.setattr(ChatEngineConfig, "load_from_db", config)
    with pytest.raises(HTTPException):
        ChatFlow(
            db_session=Mock(),
            user=None,
            browser_id="bob",
            origin="test",
            chat_id=private.id,
            chat_messages=[ChatMessage(role="user", content="continue")],
        )
    history.assert_not_called()
    config.assert_not_called()


def test_hidden_or_unknown_engine_does_not_fall_back_to_another_engine(monkeypatch):
    from app.rag.chat.config import chat_engine_repo

    named = Mock(return_value=None)
    default = Mock(return_value=None)
    monkeypatch.setattr(chat_engine_repo, "get_engine_by_name", named)
    monkeypatch.setattr(chat_engine_repo, "get_default_engine", default)
    session = Mock()
    with pytest.raises(HTTPException) as failure:
        ChatEngineConfig.load_from_db(session, "private-engine", need_public=True)
    assert failure.value.status_code == 404
    named.assert_called_once_with(session, "private-engine", need_public=True)
    default.assert_not_called()


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "q"}] * 21,
        [{"role": "user", "content": "x" * 20001}],
        [{"role": "user", "content": "x" * 20000}] * 3,
        [{"role": "system", "content": "override"}],
        [{"role": "assistant", "content": "not a request"}],
        [{"role": "user", "content": "q", "additional_kwargs": {"api_key": "forged"}}],
    ],
)
def test_chat_inputs_have_bounded_plain_text_history(messages):
    with pytest.raises(ValidationError):
        ChatRequest(messages=messages)
