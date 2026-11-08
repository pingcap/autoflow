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


def test_reading_missing_subgraph_cannot_trigger_paid_retrieval(monkeypatch):
    import pytest
    from fastapi import HTTPException

    config = Mock()
    monkeypatch.setattr(
        chat_service.ChatEngineConfig, "load_from_db", lambda *args: config
    )
    message = SimpleNamespace(
        chat=SimpleNamespace(engine=SimpleNamespace(name="default")), graph_data=None
    )
    with pytest.raises(HTTPException) as error:
        chat_service.get_chat_message_subgraph(Mock(), message)
    assert error.value.status_code == 404
    config.get_llama_llm.assert_not_called()
    config.get_knowledge_bases.assert_not_called()


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
        chat_flow,
        "external_engine_session",
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


def test_model_context_reads_bounded_recent_text_without_loading_full_transcript():
    from datetime import datetime, UTC
    from sqlmodel import create_engine, Session
    from sqlalchemy import event
    from app.repositories import chat_repo

    engine = create_engine("sqlite://")
    ChatMessage.__table__.create(engine)
    chat_id, unrelated_id = uuid4(), uuid4()
    with Session(engine) as session:
        for index in range(1, 61):
            session.add(
                ChatMessage(
                    id=index,
                    ordinal=index,
                    role="assistant" if index % 2 == 0 else "user",
                    content=f"{index}:" + "x" * 4990,
                    chat_id=chat_id,
                    finished_at=datetime.now(UTC),
                )
            )
        session.add(
            ChatMessage(
                id=61,
                ordinal=61,
                role="assistant",
                content="private unrelated transcript",
                chat_id=unrelated_id,
            )
        )
        session.commit()
        sql = []

        @event.listens_for(engine, "before_cursor_execute")
        def record(connection, cursor, statement, parameters, context, executemany):
            sql.append(statement)

        rows = chat_repo.get_context_messages(
            session, SimpleNamespace(id=chat_id), 20000
        )
        assert sum(len(content) for _, content in rows) == 20000
        assert len(rows) <= 19 and rows[-1][1].startswith("60:")
        assert rows[-2][1].startswith("59:")
        assert "private unrelated transcript" not in str(rows)
        assert "LIMIT" in sql[0] and "substr(" in sql[0]
        assert "JOIN" not in sql[0] and "graph_data" not in sql[0]
        session.add(
            ChatMessage(
                id=62,
                ordinal=62,
                role="assistant",
                content="x" * 200000 + "newest tail",
                chat_id=chat_id,
            )
        )
        session.commit()
        rows = chat_repo.get_context_messages(
            session, SimpleNamespace(id=chat_id), 10000
        )
        assert (
            len(rows) == 1
            and len(rows[0][1]) == 10000
            and rows[0][1].endswith("newest tail")
        )
    engine.dispose()


def test_get_recommendations_is_read_only_even_for_chat_owner(monkeypatch):
    from app.api.routes import chat as routes

    session = Mock()
    session.exec.return_value.first.return_value = None
    message = SimpleNamespace(
        id=1, chat=SimpleNamespace(engine=SimpleNamespace(name="default"))
    )
    request = SimpleNamespace(state=SimpleNamespace(browser_id="owner"))
    monkeypatch.setattr(routes.chat_repo, "must_get_message", lambda *args: message)
    monkeypatch.setattr(routes, "user_can_view_chat", lambda *args: True)
    monkeypatch.setattr(routes, "user_can_edit_chat", lambda *args: True)
    monkeypatch.setattr(routes, "user_can_use_engine", lambda *args: True)
    load = Mock()
    monkeypatch.setattr(chat_service.ChatEngineConfig, "load_from_db", load)
    assert routes.get_recommended_questions(request, session, None, 1) == []
    load.assert_not_called()
    session.add.assert_not_called()
    session.commit.assert_not_called()


def test_read_only_viewer_post_can_only_read_existing_recommendations(monkeypatch):
    import pytest
    from fastapi import HTTPException
    from app.api.routes import chat as routes

    session = Mock()
    session.exec.return_value.first.return_value = ["Cached question"]
    message = SimpleNamespace(
        id=1, chat=SimpleNamespace(engine=SimpleNamespace(name="default"))
    )
    request = SimpleNamespace(state=SimpleNamespace(browser_id="reader"))
    monkeypatch.setattr(routes.chat_repo, "must_get_message", lambda *args: message)
    monkeypatch.setattr(routes, "user_can_view_chat", lambda *args: True)
    monkeypatch.setattr(routes, "user_can_edit_chat", lambda *args: False)
    monkeypatch.setattr(routes, "user_can_use_engine", lambda *args: True)
    load, remove = Mock(), Mock()
    monkeypatch.setattr(chat_service.ChatEngineConfig, "load_from_db", load)
    monkeypatch.setattr(chat_service, "remove_chat_message_recommend_questions", remove)
    assert routes.refresh_recommended_questions(
        request, session, None, 1, refresh=False
    ) == ["Cached question"]
    with pytest.raises(HTTPException) as error:
        routes.refresh_recommended_questions(request, session, None, 1, refresh=True)
    assert error.value.status_code == 403
    remove.assert_not_called()
    load.assert_not_called()
