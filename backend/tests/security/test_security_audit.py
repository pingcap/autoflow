import asyncio
import json
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import event, text, inspect
from sqlmodel import Session, SQLModel, Field, Column, JSON, create_engine, select

from app.models.site_setting import SiteSetting
from app.models.staff_action_log import StaffActionLog
from app.security.audit import (
    AuditContext,
    AuditMiddleware,
    SecurityEventFormatter,
    audit_context,
    bind_actor,
    install_audit_hooks,
    security_logger,
)


@contextmanager
def actor(method="PUT", identifier=None):
    context = AuditContext(
        actor_id=str(identifier or uuid4()),
        actor_type="user",
        privileged=True,
        method=method,
    )
    token = audit_context.set(context)
    try:
        yield context
    finally:
        audit_context.reset(token)


@pytest.fixture
def database():
    engine = create_engine("sqlite://")
    SiteSetting.__table__.create(engine)
    StaffActionLog.__table__.create(engine)
    install_audit_hooks()
    yield engine
    engine.dispose()


def test_configuration_mutations_are_attributed_without_values_and_rollback_is_atomic(
    database,
):
    with Session(database) as session:
        setting = SiteSetting(
            name="langfuse_secret_key", value="private-canary", data_type="str"
        )
        with actor("POST") as first:
            session.add(setting)
            session.commit()
        with actor() as second:
            setting.value = "replacement-canary"
            session.commit()
        with actor():
            setting.value = "must-roll-back"
            session.flush()
            session.flush()
            session.rollback()
        rows = session.exec(select(StaffActionLog).order_by(StaffActionLog.id)).all()
        assert len(rows) == 2
        assert [row.actor_id for row in rows] == [first.actor_id, second.actor_id]
        assert [row.request_id for row in rows] == [first.request_id, second.request_id]
        assert rows[1].changed_fields == ["value"]
        assert rows[0].target_identifier == "langfuse_secret_key"
        assert all(row.before == row.after == {} for row in rows)
        assert "canary" not in json.dumps([row.model_dump(mode="json") for row in rows])
        assert session.exec(select(SiteSetting.value)).one() == "replacement-canary"


def test_async_session_mutations_keep_authenticated_attribution():
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel.ext.asyncio.session import AsyncSession

    async def exercise():
        engine = create_async_engine("sqlite+aiosqlite://")
        async with engine.begin() as connection:
            await connection.run_sync(SiteSetting.__table__.create)
            await connection.run_sync(StaffActionLog.__table__.create)
        install_audit_hooks()
        async with AsyncSession(engine) as session:
            with actor("POST") as principal:
                session.add(
                    SiteSetting(name="title", value="private value", data_type="str")
                )
                await session.commit()
            row = (await session.exec(select(StaffActionLog))).one()
            assert (
                row.actor_id == principal.actor_id
                and row.request_id == principal.request_id
            )
            assert "private value" not in row.model_dump_json()
        await engine.dispose()

    asyncio.run(exercise())


def test_bulk_mutations_are_audited_in_the_same_transaction(database):
    from sqlmodel import update, delete

    with Session(database) as session:
        setting = SiteSetting(name="title", value="original", data_type="str")
        session.add(setting)
        session.commit()
        identifier = setting.id
        with actor():
            session.exec(
                update(SiteSetting)
                .where(SiteSetting.id == identifier)
                .values(value="private bulk change")
            )
            session.commit()
        with actor("DELETE"):
            session.exec(delete(SiteSetting).where(SiteSetting.id == identifier))
            session.rollback()
        rows = session.exec(select(StaffActionLog)).all()
        assert len(rows) == 1 and rows[0].target_id == identifier
        assert rows[0].changed_fields == ["value"]
        assert "private" not in rows[0].model_dump_json()
        assert session.get(SiteSetting, identifier) is not None
        with actor():
            session.exec(update(SiteSetting).values(value="all"))
            session.commit()
        assert session.exec(
            select(StaffActionLog.target_identifier).order_by(StaffActionLog.id)
        ).all() == [str(identifier), "bulk"]


def test_credential_audit_supports_uuid_targets_and_omits_passwords(database):
    from app.models.auth import User
    from sqlmodel import update

    User.__table__.create(database)
    with Session(database) as session:
        user = User(email="security-user@example.test", hashed_password="old-secret")
        session.add(user)
        session.commit()
        identifier = user.id
        with actor("PATCH") as administrator:
            session.exec(
                update(User)
                .where(User.id == identifier)
                .values(hashed_password="replacement-secret")
            )
            session.commit()
        row = session.exec(select(StaffActionLog)).one()
        assert row.actor_id == administrator.actor_id
        assert row.target_identifier == str(identifier)
        assert row.target_id == 0
        assert row.changed_fields == ["hashed_password"]
        assert (
            "secret" not in row.model_dump_json() and "@" not in row.model_dump_json()
        )


def test_valid_recovery_token_attributes_change_to_its_subject_without_logging_token():
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel.ext.asyncio.session import AsyncSession
    from fastapi_users.jwt import generate_jwt
    from fastapi_users.exceptions import InvalidResetPasswordToken
    from app.auth.users import UserManager
    from app.auth.db import RevokingUserDatabase
    from app.models import User, UserSession, ApiKey

    async def exercise():
        engine = create_async_engine("sqlite+aiosqlite://")
        async with engine.begin() as connection:
            for table in (
                User.__table__,
                UserSession.__table__,
                ApiKey.__table__,
                StaffActionLog.__table__,
            ):
                await connection.run_sync(table.create)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            manager = UserManager(RevokingUserDatabase(session, User))
            user = User(
                email="recovery@example.test",
                hashed_password=manager.password_helper.hash("old-private-password"),
                is_active=True,
            )
            session.add(user)
            await session.commit()
            signed = generate_jwt(
                {
                    "sub": str(user.id),
                    "aud": manager.reset_password_token_audience,
                    "password_fgpt": manager.password_helper.hash(user.hashed_password),
                },
                manager.reset_password_token_secret,
                300,
            )
            with actor("POST") as context:
                unrelated_cookie_user = context.actor_id
                with pytest.raises(InvalidResetPasswordToken):
                    await manager.reset_password(
                        "invalid-private-token", "new-private-password"
                    )
                assert (await session.exec(select(StaffActionLog))).all() == []
                await manager.reset_password(signed, "new-private-password")
                rows = (await session.exec(select(StaffActionLog))).all()
                assert len(rows) == 1
                assert rows[0].actor_type == "recovery" and rows[0].actor_id == str(
                    user.id
                )
                assert rows[0].actor_id != unrelated_cookie_user
                assert rows[0].changed_fields == ["auth_version", "hashed_password"]
                assert (
                    "private" not in rows[0].model_dump_json()
                    and signed not in rows[0].model_dump_json()
                )
                with pytest.raises(InvalidResetPasswordToken):
                    await manager.reset_password(signed, "another-private-password")
                assert len((await session.exec(select(StaffActionLog))).all()) == 1
        await engine.dispose()

    asyncio.run(exercise())


def test_audit_write_failure_prevents_the_protected_change(database):
    with Session(database) as session:
        setting = SiteSetting(name="title", value="original", data_type="str")
        session.add(setting)
        session.commit()

        def reject(mapper, connection, target):
            raise RuntimeError("audit unavailable")

        event.listen(StaffActionLog, "before_insert", reject)
        try:
            with actor(), pytest.raises(RuntimeError, match="audit unavailable"):
                setting.value = "must not commit"
                session.commit()
            session.rollback()
        finally:
            event.remove(StaffActionLog, "before_insert", reject)
        assert session.exec(select(SiteSetting.value)).one() == "original"
        assert session.exec(select(StaffActionLog)).all() == []


class AuditTestEntity(SQLModel, table=True):
    __tablename__ = "audit_test_entities"
    id: int | None = Field(default=None, primary_key=True)
    name: str
    description: str
    description_vec: list = Field(sa_column=Column(JSON))
    meta: dict = Field(sa_column=Column(JSON))
    meta_vec: list = Field(sa_column=Column(JSON))
    entity_type: str
    synopsis_info: dict = Field(sa_column=Column(JSON))

    def screenshot(self):
        return self.model_dump()


def test_synopsis_and_kb_scoped_audit_commit_together(database, monkeypatch):
    from app.rag.indices.knowledge_graph.graph_store import tidb_graph_editor as module

    AuditTestEntity.__table__.create(database)
    monkeypatch.setattr(module, "get_entity_description_embedding", lambda *args: [0.0])
    monkeypatch.setattr(module, "get_entity_metadata_embedding", lambda *args: [0.0])
    monkeypatch.setattr(module, "TiDBGraphStore", lambda **kwargs: Mock())
    editor = module.TiDBGraphEditor.__new__(module.TiDBGraphEditor)
    editor.knowledge_base_id = 17
    editor._entity_db_model = AuditTestEntity
    editor._relationship_db_model = None
    editor._embed_model = Mock()
    with Session(database) as session, actor("POST") as principal:
        entity = editor.create_synopsis_entity(
            session,
            "private graph name",
            "private description",
            "private topic",
            {},
            [],
        )
        entity_id = entity.id
    with Session(database) as session:
        row = session.exec(select(StaffActionLog)).one()
        assert row.target_id == entity_id
        assert row.knowledge_base_id == 17 and row.actor_id == principal.actor_id
        assert row.request_id == principal.request_id
        assert row.action == "create_synopsis_entity"
        assert "private" not in row.model_dump_json()
        assert session.get(AuditTestEntity, entity_id) is not None

    def reject(mapper, connection, target):
        raise RuntimeError("audit failure")

    event.listen(StaffActionLog, "before_insert", reject)
    try:
        with Session(database) as session, actor("POST"):
            with pytest.raises(RuntimeError, match="audit failure"):
                editor.create_synopsis_entity(
                    session, "must roll back", "description", "topic", {}, []
                )
            session.rollback()
            assert len(session.exec(select(AuditTestEntity)).all()) == 1
            assert len(session.exec(select(StaffActionLog)).all()) == 1
    finally:
        event.remove(StaffActionLog, "before_insert", reject)


def test_security_events_use_route_templates_and_server_request_ids_without_credentials():
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    security_logger.addHandler(handler)
    app = FastAPI()

    @app.post("/api/v1/auth/login")
    def login():
        raise HTTPException(400, "invalid login")

    @app.put("/api/v1/admin/objects/{identifier}")
    def update(identifier):
        bind_actor(SimpleNamespace(id=uuid4(), is_superuser=True))
        return {}

    @app.get("/api/v1/private/{identifier}")
    def denied(identifier):
        raise HTTPException(403)

    app.add_middleware(AuditMiddleware)
    try:
        with TestClient(app) as client:
            failed = client.post(
                "/api/v1/auth/login?token=query-canary",
                json={"password": "body-canary"},
                headers={
                    "Authorization": "Bearer header-canary",
                    "X-Request-ID": "forged-request-id",
                },
            )
            updated = client.put("/api/v1/admin/objects/path-canary")
            client.get("/api/v1/private/path-canary")
        encoded = [SecurityEventFormatter().format(record) for record in records]
        rows = [json.loads(row) for row in encoded]
        assert [row["event"] for row in rows] == [
            "authentication_request",
            "privileged_request",
            "access_denied",
        ]
        assert rows[0]["status"] == 400 and rows[2]["status"] == 403
        assert rows[1]["route"] == "/api/v1/admin/objects/{identifier}"
        assert len({row["request_id"] for row in rows}) == 3
        assert failed.headers["x-request-id"] != "forged-request-id"
        assert updated.headers["x-request-id"] == rows[1]["request_id"]
        assert "canary" not in "".join(encoded) and "forged" not in "".join(encoded)
        assert audit_context.get() is None
    finally:
        security_logger.removeHandler(handler)


def test_rejected_api_key_records_no_raw_key():
    from app.auth.api_keys import api_key_manager

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    security_logger.addHandler(handler)
    try:
        assert (
            asyncio.run(
                api_key_manager.get_active_user_by_raw_api_key(
                    AsyncMock(), "private-invalid-key"
                )
            )
            is None
        )
        assert records[-1].security_event["event"] == "api_key_rejected"
        assert "private-invalid-key" not in SecurityEventFormatter().format(records[-1])
    finally:
        security_logger.removeHandler(handler)


def test_audit_migration_preserves_legacy_records_without_inventing_attribution():
    import importlib.util
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE staff_action_logs (id INTEGER PRIMARY KEY, action TEXT, target_type TEXT, target_id INTEGER)"
            )
        )
        connection.execute(
            text("INSERT INTO staff_action_logs VALUES (1, 'update', 'entity', 5)")
        )
        spec = importlib.util.spec_from_file_location(
            "audit_migration", "app/alembic/versions/31f7b624d9a0_attributed_audit.py"
        )
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        assert connection.execute(
            text("SELECT actor_id, knowledge_base_id, target_id FROM staff_action_logs")
        ).one() == (None, None, 5)
        assert "changed_fields" in {
            item["name"]
            for item in inspect(connection).get_columns("staff_action_logs")
        }
        migration.downgrade()
        assert (
            connection.execute(
                text("SELECT target_id FROM staff_action_logs")
            ).scalar_one()
            == 5
        )
    engine.dispose()
