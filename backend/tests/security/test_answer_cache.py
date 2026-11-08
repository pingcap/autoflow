from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlmodel import Session, SQLModel, create_engine

from app.models import Chat, ChatEngine, ChatMessage, ChatVisibility, User
from app.repositories import chat_repo
from app.rag.chat.config import ChatEngineConfig
from app.security.answer_cache import answer_cache_context, answer_goal_key


CONTEXT = "a" * 64
GOAL = answer_goal_key("goal", {"Lang": "English", "Format": "table"})
OWNER = uuid4()


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        engine,
        tables=[
            User.__table__,
            ChatEngine.__table__,
            Chat.__table__,
            ChatMessage.__table__,
        ],
    )
    with Session(engine, expire_on_commit=False) as session:
        yield session
    engine.dispose()


def current(**changes):
    return Chat(
        **{
            "title": "question",
            "engine_id": 1,
            "user_id": OWNER,
            "browser_id": "guest-a",
            "visibility": ChatVisibility.PRIVATE,
            **changes,
        }
    )


def cache_answer(session, *, chat_changes=None, message_changes=None):
    chat = current(**(chat_changes or {}))
    session.add(chat)
    question = ChatMessage(
        chat_id=chat.id,
        role="user",
        ordinal=1,
        content="question",
        user_id=chat.user_id,
    )
    answer = ChatMessage(
        **{
            "chat_id": chat.id,
            "role": "assistant",
            "ordinal": 2,
            "user_id": chat.user_id,
            "content": "private answer",
            "is_best_answer": True,
            "created_at": datetime.now(UTC),
            "finished_at": datetime.now(UTC),
            "meta": {"_answer_cache_context": CONTEXT, "_answer_cache_goal": GOAL},
            **(message_changes or {}),
        }
    )
    session.add(question)
    session.add(answer)
    session.commit()
    return answer


def lookups(session, chat):
    return [
        chat_repo.find_best_answer_for_question(
            session,
            "question",
            chat=chat,
            cache_context=CONTEXT,
        ),
        chat_repo.find_recent_assistant_messages_by_goal(
            session,
            GOAL,
            chat=chat,
            cache_context=CONTEXT,
        ),
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"user_id": uuid4()},
        {"user_id": None},
        {"api_key_id": 12},
        {"engine_id": 2},
        {"visibility": ChatVisibility.PUBLIC},
        {"deleted_at": datetime.now(UTC)},
    ],
)
def test_cache_never_reuses_other_principals_engines_or_visibility(session, changes):
    cache_answer(session, chat_changes=changes)
    assert lookups(session, current()) == [[], []]


@pytest.mark.parametrize(
    "changes",
    [
        {"meta": {}},
        {"meta": {"_answer_cache_context": "b" * 64, "_answer_cache_goal": GOAL}},
        {"finished_at": None},
        {"created_at": datetime.now(UTC) - timedelta(days=16)},
        {"ordinal": 4},
        {"is_best_answer": False},
    ],
)
def test_cache_excludes_legacy_stale_changed_and_unfinished_answers(session, changes):
    cache_answer(session, message_changes=changes)
    assert lookups(session, current()) == [[], []]


@pytest.mark.parametrize(
    "principal",
    [
        {},
        {"api_key_id": 8},
        {"user_id": None, "browser_id": "guest-a"},
    ],
)
def test_cache_preserves_same_principal_reuse_and_returns_only_newest(
    session, principal
):
    cache_answer(session, chat_changes=principal)
    newest = cache_answer(session, chat_changes=principal)
    assert lookups(session, current(**principal)) == [[newest], [newest]]


def test_guest_and_machine_credentials_cannot_reuse_each_others_answers(session):
    cache_answer(session, chat_changes={"user_id": None})
    assert lookups(session, current(user_id=None, browser_id="guest-b")) == [[], []]
    assert lookups(session, current(user_id=None, browser_id=None)) == [[], []]
    assert lookups(session, current(user_id=None, api_key_id=12)) == [[], []]
    cache_answer(session, chat_changes={"api_key_id": 8})
    assert lookups(session, current(api_key_id=9)) == [[], []]


def test_question_and_complete_format_are_part_of_matching(session):
    cache_answer(session)
    assert not chat_repo.find_best_answer_for_question(
        session,
        "different",
        chat=current(),
        cache_context=CONTEXT,
    )
    for key in [
        answer_goal_key("different goal", {"Lang": "English", "Format": "table"}),
        answer_goal_key("goal", {"Lang": "English", "Format": "paragraph"}),
    ]:
        assert not chat_repo.find_recent_assistant_messages_by_goal(
            session,
            key,
            chat=current(),
            cache_context=CONTEXT,
        )


def test_private_configuration_and_revisions_change_context_without_exposing_secrets():
    config = ChatEngineConfig()
    config._db_chat_engine = ChatEngine(id=1, name="external")
    model = SimpleNamespace(
        model_dump=Mock(return_value={"id": 2, "updated_at": "2026-01-01"})
    )
    before = answer_cache_context(config, [], [model])
    assert len(before) == 64
    config.post_verification_token = "secret-not-in-public-screenshot"
    assert answer_cache_context(config, [], [model]) != before
    assert "secret-not-in-public-screenshot" not in answer_cache_context(
        config, [], [model]
    )
    before = answer_cache_context(config, [], [model])
    model.model_dump.return_value["updated_at"] = "2026-01-02"
    assert answer_cache_context(config, [], [model]) != before
    model.model_dump.assert_called_with(mode="json", exclude={"credentials"})


@pytest.mark.parametrize(
    "enabled,history,expect_cache",
    [
        (True, [], True),
        (True, ["previous turn"], False),
        (False, [], False),
    ],
)
def test_external_flow_uses_authorized_scope_and_only_marks_first_turn_answers(
    monkeypatch,
    caplog,
    enabled,
    history,
    expect_cache,
):
    from app.rag.chat import chat_flow

    flow = chat_flow.ChatFlow.__new__(chat_flow.ChatFlow)
    flow.engine_config = ChatEngineConfig.model_validate(
        {
            "external_engine_config": {
                "stream_chat_api_url": "https://engine.example.test/private-endpoint"
            },
        }
    )
    flow.engine_config._db_chat_engine = ChatEngine(id=1, name="external")
    flow.db_chat_obj = current()
    flow.db_session = Mock()
    flow.chat_history = history
    flow.knowledge_bases = []
    flow.user_question = "private-question-marker"
    user = ChatMessage(id=1, role="user", content=flow.user_question)
    answer = ChatMessage(id=2, role="assistant", content="")

    def start():
        yield from ()
        return user, answer

    def goal():
        yield from ()
        return "private-goal-marker", {"Lang": "English"}

    flow._chat_start = start
    flow._generate_goal = goal
    flow._post_verification = Mock(return_value=None)
    monkeypatch.setattr(chat_flow.settings, "ENABLE_QUESTION_CACHE", enabled)
    monkeypatch.setattr(chat_flow.llm_repo, "get_default", Mock(return_value=None))
    monkeypatch.setattr(
        chat_flow.reranker_model_repo, "get_default", Mock(return_value=None)
    )
    question_lookup, goal_lookup = Mock(return_value=[]), Mock(return_value=[])
    monkeypatch.setattr(
        chat_flow.chat_repo, "find_best_answer_for_question", question_lookup
    )
    monkeypatch.setattr(
        chat_flow.chat_repo, "find_recent_assistant_messages_by_goal", goal_lookup
    )
    response = Mock()
    response.iter_lines.return_value = [b'0:"private-answer-marker"']
    monkeypatch.setattr(chat_flow.requests, "post", Mock(return_value=response))

    with caplog.at_level("DEBUG", logger="app.rag.chat.chat_flow"):
        list(flow._external_chat())
    assert question_lookup.called == expect_cache
    assert goal_lookup.called == expect_cache
    assert ("_answer_cache_context" in answer.meta) == expect_cache
    if expect_cache:
        for lookup in [question_lookup, goal_lookup]:
            assert lookup.call_args.kwargs["chat"] is flow.db_chat_obj
            assert (
                lookup.call_args.kwargs["cache_context"]
                == answer.meta["_answer_cache_context"]
            )
    flow.db_session.commit.assert_called_once()
    assert answer.content == "private-answer-marker"
    for marker in [
        "private-question-marker",
        "private-goal-marker",
        "private-answer-marker",
        "private-endpoint",
    ]:
        assert marker not in caplog.text
