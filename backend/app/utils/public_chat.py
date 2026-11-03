"""Explicit public projections for stored chat configuration and sources."""

import json


def _options(value) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {"hide_sources": True}
    return value if isinstance(value, dict) else {"hide_sources": True}


def public_engine_options(value) -> dict:
    options = _options(value)
    hidden = _hidden(options)
    graph = _options(options.get("knowledge_graph", {}))
    external = _options(options.get("external_engine_config", {}))
    return {
        "hide_sources": hidden,
        "further_questions": options.get("further_questions", True) is not False,
        "clarify_question": options.get("clarify_question", False) is True,
        "knowledge_graph": {
            "enabled": graph.get("enabled", True) is not False and not hidden
        },
        "external_engine_config": {
            "enabled": bool(
                external.get("enabled") or external.get("stream_chat_api_url")
            )
        },
    }


def chat_hides_sources(chat) -> bool:
    # Use the current engine policy when the ORM already joined it, including
    # chats created before an operator enabled hide_sources. Never trigger I/O
    # from a serializer after its request session has closed.
    engine = chat.__dict__.get("engine")
    options = engine.engine_options if engine is not None else chat.engine_options
    return _hidden(_options(options))


def _hidden(options: dict) -> bool:
    value = options.get("hide_sources", False)
    if value is False or value == 0:
        return False
    if isinstance(value, str) and value.strip().lower() in {
        "false",
        "0",
        "off",
        "no",
        "",
    }:
        return False
    return True


def public_chat_message(message, hide_sources: bool):
    if not hide_sources:
        return message
    return message.model_copy(update={"sources": [], "graph_data": None})
