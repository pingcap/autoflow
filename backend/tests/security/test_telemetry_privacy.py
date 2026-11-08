import json
from types import SimpleNamespace

import pytest
from sentry_sdk import Client, Scope
from sentry_sdk.scope import use_scope
from sentry_sdk.attachments import Attachment
from sentry_sdk.transport import Transport

from app.security.telemetry import privacy_options


PRIVATE = "private-content-marker"


class MemoryTransport(Transport):
    def __init__(self, options):
        super().__init__(options)
        self.envelopes = []

    def capture_envelope(self, envelope):
        self.envelopes.append(envelope)


def client():
    return Client(
        dsn="https://public@example.invalid/1",
        transport=MemoryTransport,
        default_integrations=False,
        traces_sample_rate=1.0,
        **privacy_options([SimpleNamespace(path="/api/v1/chats/{chat_id}")]),
    )


def wire(client):
    return b"\n".join(envelope.serialize() for envelope in client.transport.envelopes)


def private_event(**changes):
    return {
        "message": PRIVATE,
        "logentry": {"message": PRIVATE, "params": [PRIVATE]},
        "user": {"id": PRIVATE, "email": PRIVATE, "ip_address": "192.0.2.1"},
        "request": {
            "method": "POST",
            "url": "https://example.invalid/" + PRIVATE,
            "data": PRIVATE,
            "query_string": PRIVATE,
            "headers": {"Authorization": PRIVATE, "Cookie": PRIVATE},
        },
        "extra": {"input": PRIVATE},
        "tags": {"context": PRIVATE},
        "breadcrumbs": {"values": [{"message": PRIVATE, "data": {"query": PRIVATE}}]},
        "contexts": {
            "custom": {"input": PRIVATE},
            "trace": {
                "trace_id": "a" * 32,
                "span_id": "b" * 16,
                "op": "http.server",
                "data": {"prompt": PRIVATE},
                "dynamic_sampling_context": {"transaction": PRIVATE},
            },
        },
        "transaction": "/api/v1/chats/{chat_id}",
        **changes,
    }


def test_actual_sdk_event_envelope_omits_all_application_data_and_attachments():
    sdk = client()
    try:
        event_id = sdk.capture_event(
            private_event(),
            hint={
                "attachments": [
                    Attachment(bytes=PRIVATE.encode(), filename="context.txt")
                ],
            },
        )
        assert event_id
        output = wire(sdk)
        assert PRIVATE.encode() not in output
        assert b"192.0.2.1" not in output
        assert b'"type":"attachment"' not in output
        event = sdk.transport.envelopes[0].items[0].payload.json
        assert event["request"] == {"method": "POST"}
        assert event["transaction"] == "/api/v1/chats/{chat_id}"
        assert event["contexts"]["trace"]["trace_id"] == "a" * 32
    finally:
        sdk.close()


def test_actual_sdk_exception_retains_type_and_code_location_without_message_or_locals():
    sdk = client()
    scope = Scope(client=sdk)
    try:
        try:
            private_local = PRIVATE
            raise ValueError(private_local)
        except ValueError as error:
            with use_scope(scope):
                assert scope.capture_exception(error)
        assert PRIVATE.encode() not in wire(sdk)
        event = sdk.transport.envelopes[0].items[0].payload.json
        error = event["exception"]["values"][-1]
        assert error["type"] == "ValueError"
        assert (
            error["stacktrace"]["frames"][-1]["filename"] == "test_telemetry_privacy.py"
        )
        assert error["stacktrace"]["frames"][-1]["lineno"] > 0
        assert "vars" not in json.dumps(event)
    finally:
        sdk.close()


def test_actual_sdk_transaction_envelope_omits_span_sql_urls_and_unknown_route():
    sdk = client()
    try:
        event = private_event(
            type="transaction",
            transaction="/missing/" + PRIVATE,
            start_timestamp=1.0,
            timestamp=2.0,
            spans=[
                {
                    "trace_id": "a" * 32,
                    "span_id": "c" * 16,
                    "parent_span_id": "b" * 16,
                    "op": "db.sql.query",
                    "start_timestamp": 1.1,
                    "timestamp": 1.9,
                    "description": PRIVATE,
                    "data": {"sql.params": PRIVATE},
                }
            ],
        )
        assert sdk.capture_event(event)
        assert PRIVATE.encode() not in wire(sdk)
        result = sdk.transport.envelopes[0].items[0].payload.json
        assert result["transaction"] == "application"
        assert result["timestamp"] - result["start_timestamp"] == 1
        assert result["spans"][0]["op"] == "db.sql.query"
        assert result["spans"][0]["timestamp"] == 1.9
    finally:
        sdk.close()


@pytest.mark.parametrize(
    "invalid",
    [
        {"request": None},
        {"exception": {"values": None}},
    ],
)
def test_malformed_events_are_dropped_instead_of_bypassing_projection(invalid):
    sdk = client()
    try:
        assert sdk.capture_event(private_event(**invalid)) is None
        assert not sdk.transport.envelopes
    finally:
        sdk.close()


def test_other_sdk_collection_channels_are_disabled_and_breadcrumbs_are_dropped():
    options = privacy_options()
    assert options["before_breadcrumb"]({"message": PRIVATE}, {}) is None
    assert options["include_local_variables"] is False
    assert options["include_source_context"] is False
    assert options["max_request_body_size"] == "never"
    assert options["send_default_pii"] is False
    assert (
        options["profiles_sample_rate"] == options["profile_session_sample_rate"] == 0
    )
    assert options["auto_session_tracking"] is False
    assert not any(options["_experiments"].values())


def test_nested_fastapi_routes_keep_templates_but_never_actual_request_paths():
    from fastapi import FastAPI, APIRouter
    from fastapi.routing import iter_route_contexts

    router = APIRouter()

    @router.get("/chats/{chat_id}")
    def chat(chat_id: str):
        return {}

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    scrub = privacy_options(iter_route_contexts(app.routes))["before_send"]
    assert scrub(private_event(), {})["transaction"] == "/api/v1/chats/{chat_id}"
    assert (
        scrub(private_event(transaction="/api/v1/chats/" + PRIVATE), {})["transaction"]
        == "application"
    )
