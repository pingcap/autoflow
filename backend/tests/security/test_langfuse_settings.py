import base64
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from threading import Barrier, Thread
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest
from sqlalchemy import event
from sqlmodel import Session, create_engine, select
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.models.base import get_aes_key
from app.models.site_setting import SiteSetting
from app.security.langfuse_settings import (
    CONFIG_NAME,
    DEFAULT_HOST,
    LEGACY_NAMES,
    ConfigurationConflict,
    LangfuseConfigurationError,
    LangfuseUpdate,
    RuntimeConfiguration,
    read_public_configuration,
    read_runtime_configuration,
    save_configuration,
)
from migrate_langfuse_credentials import migrate_session

CANARY = "langfuse-private-fixture-canary"


@pytest.fixture
def database():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SiteSetting.__table__.create(engine)
    yield engine
    engine.dispose()


def replacement(session, *, secret=CANARY, **changes):
    view = read_public_configuration(session)
    return LangfuseUpdate(
        **{
            **{
                "host": view["host"],
                "public_key": "pk-fixture",
                "revision": view["revision"],
                "secret_key": SecretStr(secret) if secret is not None else None,
            },
            **changes,
        }
    )


def legacy(session):
    for name, value in [
        ("langfuse_host", DEFAULT_HOST),
        ("langfuse_public_key", "pk-fixture"),
        ("langfuse_secret_key", CANARY),
    ]:
        session.add(SiteSetting(name=name, value=value, data_type="str"))
    session.commit()


def test_ciphertext_and_public_views_do_not_return_secret(database):
    with Session(database) as session:
        result = save_configuration(session, replacement(session))
        session.commit()
        stored = session.exec(
            select(SiteSetting.value).where(SiteSetting.name == CONFIG_NAME)
        ).one()
        assert CANARY not in json.dumps(stored)
        assert CANARY not in json.dumps(result)
        assert read_public_configuration(session)["configured"]
        runtime = read_runtime_configuration(session)
        assert runtime.secret_key == CANARY and runtime.enabled
        assert CANARY not in repr(runtime)


def test_destination_changes_require_fresh_credentials_and_stale_edit_is_rejected(
    database, monkeypatch
):
    monkeypatch.setattr(
        settings, "LANGFUSE_ALLOWED_ORIGINS", ["https://langfuse.example.test"]
    )
    with Session(database) as session:
        initial = save_configuration(session, replacement(session))
        session.commit()
        with pytest.raises(LangfuseConfigurationError, match="fresh credentials"):
            save_configuration(
                session,
                replacement(session, secret=None, host="https://langfuse.example.test"),
            )
        session.rollback()
        stale = replacement(session, secret=None, revision=initial["revision"])
        changed = save_configuration(
            session,
            replacement(
                session,
                secret="replacement-secret",
                host="https://langfuse.example.test",
            ),
        )
        session.commit()
        with pytest.raises(ConfigurationConflict):
            save_configuration(session, stale)
        session.rollback()
        assert read_public_configuration(session) == changed
        assert read_runtime_configuration(session).secret_key == "replacement-secret"


@pytest.mark.parametrize(
    "field,value",
    [("host", "https://cloud.langfuse.com"), ("public_key", "different-project")],
)
def test_ciphertext_authenticates_destination(database, field, value):
    with Session(database) as session:
        save_configuration(session, replacement(session))
        session.commit()
        row = session.exec(
            select(SiteSetting).where(SiteSetting.name == CONFIG_NAME)
        ).one()
        row.value = {**row.value, field: value}
        session.add(row)
        session.commit()
        with pytest.raises(LangfuseConfigurationError, match="authentication failed"):
            read_runtime_configuration(session)


def test_migration_is_read_only_by_default_atomic_idempotent_and_supports_rotation(
    database,
):
    with Session(database) as session:
        legacy(session)
        assert read_public_configuration(session)["migration_required"]
        with pytest.raises(LangfuseConfigurationError, match="migration"):
            read_runtime_configuration(session)
        statements = []

        def record(_conn, _cursor, statement, *_args):
            statements.append(statement)

        event.listen(database, "before_cursor_execute", record)
        assert migrate_session(session, get_aes_key()) == {
            "needs_migration": 1,
            "migrated": 0,
        }
        event.remove(database, "before_cursor_execute", record)
        assert all(
            statement.lstrip().upper().startswith("SELECT") for statement in statements
        )
        migrate_session(session, get_aes_key(), apply=True)
        session.rollback()
        assert (
            session.exec(
                select(SiteSetting.value).where(
                    SiteSetting.name == "langfuse_secret_key"
                )
            ).one()
            == CANARY
        )
        assert (
            session.exec(
                select(SiteSetting).where(SiteSetting.name == CONFIG_NAME)
            ).first()
            is None
        )
        migrate_session(session, get_aes_key(), apply=True)
        session.commit()
        assert (
            session.exec(
                select(SiteSetting.value).where(
                    SiteSetting.name == "langfuse_secret_key"
                )
            ).one()
            == ""
        )
        assert read_runtime_configuration(session).secret_key == CANARY
        assert migrate_session(session, get_aes_key(), apply=True)["migrated"] == 0
        old_key = get_aes_key()
        new_key = b"n" * 32
        assert (
            migrate_session(session, new_key, previous_key=old_key, apply=True)[
                "migrated"
            ]
            == 1
        )
        session.commit()
        from app.security.langfuse_settings import decrypt_secret

        row = session.exec(
            select(SiteSetting).where(SiteSetting.name == CONFIG_NAME)
        ).one()
        assert decrypt_secret(row.value, key=new_key) == CANARY
        with pytest.raises(LangfuseConfigurationError):
            decrypt_secret(row.value, key=old_key)


def test_public_site_settings_omit_legacy_and_encrypted_secrets(database, monkeypatch):
    from app.site_settings import SiteSettingProxy

    proxy = SiteSettingProxy()
    monkeypatch.setattr(proxy, "update_db_cache", lambda *args: None)
    proxy._SiteSettingProxy__db_cache = {"langfuse_secret_key": CANARY}
    assert CANARY not in json.dumps(
        {k: asdict(v) for k, v in proxy.get_all_settings().items()}
    )
    with Session(database) as session:
        for name in LEGACY_NAMES:
            with pytest.raises(ValueError, match="together"):
                proxy.update_setting(session, name, "replacement")


@pytest.fixture
def client(database):
    from app.api.admin_routes.langfuse import router
    from app.api.deps import current_superuser, get_db_session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[current_superuser] = lambda: SimpleNamespace(
        is_superuser=True
    )

    def db_session():
        with Session(database) as session:
            yield session

    app.dependency_overrides[get_db_session] = db_session
    with TestClient(app) as client:
        yield client


def test_admin_roundtrip_secret_removal_and_safe_validation(client):
    view = client.get("/admin/langfuse").json()
    body = {
        "host": DEFAULT_HOST,
        "public_key": "pk-fixture",
        "secret_key": CANARY,
        "revision": view["revision"],
    }
    response = client.put("/admin/langfuse", json=body)
    assert response.status_code == 200 and CANARY not in response.text
    assert CANARY not in client.get("/admin/langfuse").text
    assert client.put("/admin/langfuse", json=body).status_code == 409
    invalid = client.put(
        "/admin/langfuse", json={**body, "secret_key": {"bad": CANARY}}
    )
    assert invalid.status_code == 422 and CANARY not in invalid.text
    view = response.json()
    cleared = client.put(
        "/admin/langfuse",
        json={
            "host": view["host"],
            "public_key": view["public_key"],
            "revision": view["revision"],
            "clear_secret": True,
        },
    )
    assert cleared.status_code == 200 and not cleared.json()["configured"]


@contextmanager
def server():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization"), None))
            if self.path.startswith("/error"):
                self.send_response(401)
                self.end_headers()
                self.wfile.write(CANARY.encode())
                return
            if self.path.startswith("/redirect"):
                self.send_response(302)
                self.send_header("Location", "/stolen")
                self.end_headers()
                return
            payload = {"data": [{"id": "fixture", "name": "Fixture project"}]}
            body = json.dumps(payload).encode()
            if self.path.startswith("/large"):
                body = b"x" * 65537
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(
                (self.path, self.headers.get("Authorization"), json.loads(body))
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(2)


def test_auth_test_uses_fresh_explicit_credentials_and_blocks_redirects_and_oversize(
    client, monkeypatch
):
    with server() as (host, calls):
        monkeypatch.setattr(settings, "LANGFUSE_ALLOWED_ORIGINS", [host])
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        body = {"host": host, "public_key": "pk-fixture", "secret_key": CANARY}
        result = client.post("/admin/langfuse/test", json=body)
        assert result.json()["success"]
        expected = "Basic " + base64.b64encode(f"pk-fixture:{CANARY}".encode()).decode()
        assert calls[0][1] == expected
        for path in ["/redirect", "/large", "/error"]:
            result = client.post(
                "/admin/langfuse/test", json={**body, "host": host + path}
            )
            assert not result.json()["success"] and CANARY not in result.text
        assert all(path != "/stolen" for path, *_ in calls)
        count = len(calls)
        result = client.post(
            "/admin/langfuse/test", json={**body, "host": "http://169.254.169.254"}
        )
        assert not result.json()["success"] and len(calls) == count


def test_actual_sdk_clients_route_concurrent_traces_to_their_own_destination(
    monkeypatch,
):
    from app.utils.langfuse_instrumentor import BoundLangfuseInstrumentor
    from app.utils.tracing import LangfuseContextManager
    from llama_index.core.instrumentation import get_dispatcher

    dispatcher = get_dispatcher("autoflow-security-langfuse-test")
    barrier = Barrier(2)
    marker = logging.NullHandler()
    logging.getLogger("httpx").addHandler(marker)

    @dispatcher.span
    def operation(value):
        from llama_index.core.llms import MockLLM

        barrier.wait(timeout=5)
        return MockLLM(max_tokens=2).complete(value).text

    with server() as (first, calls_a), server() as (second, calls_b):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "ambient-must-not-be-used")

        def trace(host, key, content):
            config = RuntimeConfiguration(host, "same-public-key", key, content)
            instrumentor = BoundLangfuseInstrumentor(config)
            manager = LangfuseContextManager(instrumentor)
            with manager.observe(trace_name="fixture") as trace:
                trace.update(input=content, output=operation(content))
            assert instrumentor._http.is_closed
            return instrumentor

        with ThreadPoolExecutor(max_workers=2) as executor:
            a = executor.submit(trace, first, "secret-a", "content-a")
            b = executor.submit(trace, second, "secret-b", "content-b")
            assert a.result().client_instance is not b.result().client_instance
        for calls, key, present, absent in [
            (calls_a, "secret-a", "content-a", "content-b"),
            (calls_b, "secret-b", "content-b", "content-a"),
        ]:
            assert calls
            expected = (
                "Basic " + base64.b64encode(f"same-public-key:{key}".encode()).decode()
            )
            assert all(auth == expected for _, auth, _ in calls)
            bodies = json.dumps([body for _, _, body in calls])
            assert present in bodies and absent not in bodies
            assert "operation" in bodies
            assert "generation-update" in bodies
        assert marker in logging.getLogger("httpx").handlers
    logging.getLogger("httpx").removeHandler(marker)


@pytest.mark.parametrize(
    "host",
    [
        "https://user:secret@cloud.langfuse.com",
        "file:///secret",
        "https://cloud.langfuse.com?token=private",
        "http://169.254.169.254",
        "https://cloud.langfuse.com.attacker.test",
    ],
)
def test_invalid_or_unregistered_destination_cannot_be_saved(client, host):
    view = client.get("/admin/langfuse").json()
    response = client.put(
        "/admin/langfuse",
        json={
            "host": host,
            "public_key": "pk",
            "secret_key": CANARY,
            "revision": view["revision"],
        },
    )
    assert response.status_code == 400
    assert CANARY not in response.text and host not in response.text


def test_disabled_sdk_never_uses_ambient_configuration(monkeypatch):
    from app.utils.langfuse_instrumentor import BoundLangfuseInstrumentor
    from app.utils.tracing import LangfuseContextManager

    with server() as (host, calls):
        monkeypatch.setenv("LANGFUSE_HOST", host)
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", CANARY)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "ambient-public")
        instrumentor = BoundLangfuseInstrumentor(
            RuntimeConfiguration(host, "", "", "disabled")
        )
        manager = LangfuseContextManager(instrumentor)
        with pytest.raises(RuntimeError, match="fixture"):
            with manager.observe(trace_name="fixture") as trace:
                trace.update(input="private-input")
                assert manager.trace_url is None
                raise RuntimeError("fixture")
        assert not calls
        assert instrumentor._http.is_closed
        assert instrumentor._tokens is None


def test_streaming_trace_context_survives_worker_changes_and_disconnect():
    from contextlib import nullcontext
    from app.security.generation import GenerationIterator
    from app.utils.langfuse_instrumentor import BoundLangfuseInstrumentor
    from app.utils.tracing import LangfuseContextManager

    with server() as (host, calls):
        instrumentor = BoundLangfuseInstrumentor(
            RuntimeConfiguration(host, "pk", "key", "revision")
        )
        manager = LangfuseContextManager(instrumentor)

        def stream():
            with manager.observe(trace_name="stream") as trace:
                trace.update(input="first-part")
                yield 1
                trace.update(output="second-part")
                yield 2

        iterator = GenerationIterator(
            stream(), SimpleNamespace(activate=nullcontext, close=lambda: None)
        )
        with (
            ThreadPoolExecutor(max_workers=1) as first,
            ThreadPoolExecutor(max_workers=1) as second,
        ):
            assert first.submit(next, iterator).result() == 1
            assert second.submit(next, iterator).result() == 2
            first.submit(iterator.close).result()
        assert instrumentor._http.is_closed and instrumentor._tokens is None
        bodies = json.dumps(calls)
        assert "first-part" in bodies and "second-part" in bodies
