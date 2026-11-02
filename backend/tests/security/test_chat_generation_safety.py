from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from app.models import Chat, ChatMessage
from app.rag.chat import chat_flow, chat_service


def test_public_reader_cache_miss_cannot_initialize_or_call_a_model(monkeypatch):
    session = Mock()
    session.exec.return_value.first.return_value = None
    load = Mock()
    monkeypatch.setattr(chat_service.ChatEngineConfig, "load_from_db", load)
    assert (
        chat_service.get_chat_message_recommend_questions(
            session, SimpleNamespace(id=1)
        )
        == []
    )
    load.assert_not_called()
    session.add.assert_not_called()


def test_public_reader_can_read_cached_recommendations_without_model_access(
    monkeypatch,
):
    session = Mock()
    session.exec.return_value.first.return_value = ["Previously generated question"]
    load = Mock()
    monkeypatch.setattr(chat_service.ChatEngineConfig, "load_from_db", load)
    assert chat_service.get_chat_message_recommend_questions(
        session, SimpleNamespace(id=1)
    ) == ["Previously generated question"]
    load.assert_not_called()


def test_post_verification_failure_without_message_attribute_does_not_lose_chat(
    monkeypatch, caplog
):
    flow = chat_flow.ChatFlow.__new__(chat_flow.ChatFlow)
    flow.engine_config = SimpleNamespace(
        post_verification_url="https://verify.example.test",
        post_verification_token="private-token",
    )
    flow._trace_manager = Mock()
    flow._trace_manager.span.return_value = nullcontext(Mock())
    flow.db_chat_obj = Chat(
        id=uuid4(), title="q", engine_options={"hide_sources": True}
    )
    flow.db_session = Mock()
    flow.user_question = "question"
    monkeypatch.setattr(
        chat_flow.requests,
        "post",
        Mock(side_effect=RuntimeError("private-token must never be logged")),
    )
    user_message = ChatMessage(id=1, role="user", content="question")
    assistant_message = ChatMessage(id=2, role="assistant", content="")
    events = list(
        flow._chat_finish(assistant_message, user_message, "completed answer")
    )
    assert len(events) == 2
    flow.db_session.commit.assert_called_once()
    assert assistant_message.content == "completed answer"
    assert assistant_message.post_verification_result_url is None
    assert "private-token" not in caplog.text
