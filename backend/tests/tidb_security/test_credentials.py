from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import json

from fastapi import HTTPException
from pydantic import SecretStr
import pytest
from sqlalchemy import event, text
from sqlmodel import Session, select

from app.core.config import settings
from app.models import LLM, LLMUpdate
from app.models.base import get_aes_key
from app.models.site_setting import SiteSetting
from app.rag.llms.provider import LLMProvider
from app.repositories.llm import llm_repo
from app.security.langfuse_settings import (
    CONFIG_NAME,
    DEFAULT_HOST,
    ConfigurationConflict,
    LangfuseUpdate,
    read_public_configuration,
    read_runtime_configuration,
    save_configuration,
)
from migrate_langfuse_credentials import migrate_session


def replacement(session, **changes):
    current = read_public_configuration(session)
    return LangfuseUpdate(
        **{
            "host": current["host"],
            "public_key": "pk-fixture",
            "revision": current["revision"],
            "secret_key": SecretStr("fixture-key"),
            **changes,
        }
    )


def test_tidb_json_compare_and_swap_preserves_ciphertext_and_rollback(database):
    with Session(database) as session:
        save_configuration(session, replacement(session))
        session.commit()
        initial = read_public_configuration(session)
        changed = save_configuration(
            session, replacement(session, secret_key=SecretStr("new-fixture-key"))
        )
        session.commit()
        assert changed["revision"] != initial["revision"]
        assert read_runtime_configuration(session).secret_key == "new-fixture-key"
        raw = session.exec(
            select(SiteSetting.value).where(SiteSetting.name == CONFIG_NAME)
        ).one()
        assert "fixture-key" not in json.dumps(raw)
        save_configuration(
            session, replacement(session, clear_secret=True, secret_key=None)
        )
        session.rollback()
        assert read_runtime_configuration(session).secret_key == "new-fixture-key"


def test_tidb_competing_editors_cannot_overwrite_newer_revision(database):
    with Session(database) as session:
        save_configuration(session, replacement(session))
        session.commit()
        edit = replacement(session)
    ready = Barrier(2)

    def write(value):
        with Session(database) as session:
            ready.wait(timeout=5)
            try:
                save_configuration(
                    session, edit.model_copy(update={"secret_key": SecretStr(value)})
                )
                session.commit()
                return value
            except ConfigurationConflict:
                session.rollback()
                return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, ["fixture-one", "fixture-two"]))
    assert results.count("conflict") == 1
    with Session(database) as session:
        assert read_runtime_configuration(session).secret_key == next(
            value for value in results if value != "conflict"
        )


def test_tidb_provider_update_refreshes_stale_credentials_under_row_lock(
    database, monkeypatch
):
    monkeypatch.setattr(
        settings, "PROVIDER_ALLOWED_ORIGINS", ["https://gateway.example.test"]
    )
    with Session(database) as session:
        row = llm_repo.create(
            session,
            LLM(
                name="fixture",
                provider=LLMProvider.OPENAI,
                model="gpt-4o",
                config={},
                credentials="old-fixture",
            ),
        )
        model_id = row.id
    with Session(database) as stale, Session(database) as current:
        old = stale.get(LLM, model_id)
        new = current.get(LLM, model_id)
        llm_repo.update(
            current,
            new,
            LLMUpdate(
                config={"api_base": "https://gateway.example.test/v1"},
                credentials="new-fixture",
            ),
        )
        with pytest.raises(HTTPException, match="fresh credentials"):
            llm_repo.update(
                stale, old, LLMUpdate(config={"api_base": "https://api.openai.com/v1"})
            )
        stale.rollback()
        current.refresh(new)
        assert new.credentials == "new-fixture"
        ciphertext = current.execute(
            text("SELECT credentials FROM llms WHERE id=:id"), {"id": model_id}
        ).scalar_one()
        assert b"new-fixture" not in ciphertext


def test_tidb_legacy_migration_is_read_only_then_atomic_and_idempotent(database):
    with Session(database) as session:
        for name, value in [
            ("langfuse_host", DEFAULT_HOST),
            ("langfuse_public_key", "pk-fixture"),
            ("langfuse_secret_key", "legacy-fixture"),
        ]:
            session.add(SiteSetting(name=name, value=value, data_type="str"))
        session.commit()
        statements = []

        def read_only(_connection, _cursor, statement, *_args):
            statements.append(statement)
            assert statement.lstrip().upper().startswith("SELECT")

        event.listen(database, "before_cursor_execute", read_only)
        try:
            assert migrate_session(session, get_aes_key()) == {
                "needs_migration": 1,
                "migrated": 0,
            }
            assert statements
        finally:
            event.remove(database, "before_cursor_execute", read_only)
        assert migrate_session(session, get_aes_key(), apply=True)["migrated"] == 1
        session.rollback()
        assert read_public_configuration(session)["migration_required"]
        migrate_session(session, get_aes_key(), apply=True)
        session.commit()
        assert read_runtime_configuration(session).secret_key == "legacy-fixture"
        assert (
            session.exec(
                select(SiteSetting.value).where(
                    SiteSetting.name == "langfuse_secret_key"
                )
            ).one()
            == ""
        )
        assert migrate_session(session, get_aes_key(), apply=True) == {
            "needs_migration": 0,
            "migrated": 0,
        }
