import json

from app.models import Chat, ChatMessage, ChatEngine
from app.rag.chat.stream_protocol import ChatStreamDataPayload
from app.utils.public_chat import public_engine_options, chat_hides_sources


PRIVATE_OPTIONS = {
    "llm": {"text_qa_prompt": "operator-only prompt"},
    "knowledge_base": {"linked_knowledge_bases": [{"id": 7654}]},
    "post_verification_url": "https://internal.example/verify",
    "post_verification_token": "operator-private-token",
    "external_engine_config": {"stream_chat_api_url": "https://internal.example/chat"},
    "hide_sources": True,
    "knowledge_graph": {"enabled": True},
}


def test_public_projection_preserves_feature_flags_without_operator_configuration():
    public = public_engine_options(PRIVATE_OPTIONS)
    assert public["external_engine_config"] == {"enabled": True}
    assert public["knowledge_graph"]["enabled"] is False
    assert public["hide_sources"] is True
    serialized = json.dumps(public)
    for sensitive in ["internal.example", "operator-only", "operator-private", "7654"]:
        assert sensitive not in serialized
    assert public_engine_options(json.dumps(PRIVATE_OPTIONS)) == public


def test_legacy_chat_json_is_redacted_without_changing_persisted_state():
    chat = Chat(
        title="q", browser_id="opaque-owner-id", engine_options=PRIVATE_OPTIONS.copy()
    )
    public = chat.model_dump(mode="json")
    assert public["browser_id"] is None
    assert "internal.example" not in json.dumps(public)
    assert chat.browser_id == "opaque-owner-id"
    assert chat.model_dump()["engine_options"] == PRIVATE_OPTIONS


def test_latest_engine_privacy_applies_to_both_messages_in_stream_and_old_chats():
    chat = Chat(title="q", browser_id="owner", engine_options={"hide_sources": False})
    chat.engine = ChatEngine(name="engine", engine_options={"hide_sources": "true"})
    user = ChatMessage(
        role="user",
        content="q",
        sources=[{"id": 7654}],
        graph_data={"private": "graph"},
    )
    assistant = ChatMessage(
        role="assistant",
        content="answer",
        sources=[{"id": 7654}],
        graph_data={"private": "graph"},
    )
    assert chat_hides_sources(chat)
    public = ChatStreamDataPayload(chat, user, assistant).dump()[0]
    for key in ["user_message", "assistant_message"]:
        assert public[key]["sources"] == []
        assert public[key]["graph_data"] is None
    assert public["chat"]["engine_options"]["hide_sources"] is True
    assert assistant.sources == [{"id": 7654}]
    assert user.graph_data == {"private": "graph"}


def test_invalid_stored_options_fail_closed_for_source_visibility():
    assert public_engine_options("invalid json")["hide_sources"] is True
    assert public_engine_options({"hide_sources": None})["hide_sources"] is True
    assert public_engine_options({"hide_sources": "false"})["hide_sources"] is False
