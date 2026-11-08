"""Upgrade populated tables from the observed production schema, using fake data."""

import json
import os
import subprocess
import sys
from uuid import uuid4

import pytest
from sqlalchemy import MetaData, Table, text
from sqlmodel import Session

from app.models.base import get_aes_key
from app.security.langfuse_settings import read_runtime_configuration
from app.utils.aes import AESCipher, MAGIC
from migrate_langfuse_credentials import migrate_session
from migrate_provider_credentials import migrate_connection
from tests.security.test_credential_encryption import legacy_encrypt


def migrate(database, direction, revision):
    subprocess.run(
        [sys.executable, "-m", "alembic", direction, revision],
        env=dict(os.environ, TIDB_DATABASE=database.url.database),
        check=True,
        timeout=60,
    )


@pytest.mark.parametrize("database", ["04947f9684ab"], indirect=True)
def test_populated_production_revision_upgrade_and_safe_schema_rollback(database):
    metadata = MetaData()
    tables = {
        name: Table(name, metadata, autoload_with=database)
        for name in (
            "llms",
            "embedding_models",
            "reranker_models",
            "users",
            "user_sessions",
            "api_keys",
            "chat_engines",
            "chats",
            "site_settings",
            "staff_action_logs",
        )
    }
    owner, guest_chat, own_chat = (uuid4().hex for _ in range(3))
    legacy_key = os.urandom(48)
    current_key = get_aes_key()
    providers = ("llms", "embedding_models", "reranker_models")
    original = {}
    with database.begin() as connection:
        for name in providers:
            original[name] = legacy_encrypt({"api_key": "fixture-" + name}, legacy_key)
            extra = {"vector_dimension": 3} if name == "embedding_models" else {}
            if name == "reranker_models":
                extra["top_n"] = 10
            connection.execute(
                tables[name]
                .insert()
                .values(
                    id=1,
                    name="legacy fixture",
                    provider="openai",
                    model="fixture",
                    config={},
                    credentials=original[name],
                    is_default=True,
                    **extra,
                )
            )
        connection.execute(
            tables["users"]
            .insert()
            .values(
                id=owner,
                email="rollout@example.test",
                hashed_password="fixture",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        connection.execute(
            tables["user_sessions"]
            .insert()
            .values(
                token="fixture-existing-session",
                user_id=owner,
            )
        )
        connection.execute(
            tables["api_keys"]
            .insert()
            .values(
                id=1,
                user_id=owner,
                description="legacy key",
                api_key_display="fixture",
                hashed_secret="fixture-secret-hash",
                is_active=True,
            )
        )
        connection.execute(
            tables["chat_engines"]
            .insert()
            .values(
                id=1,
                name="fixture",
                llm_id=1,
                engine_options={},
                is_default=True,
                is_public=True,
            )
        )
        for chat_id, user in [(guest_chat, None), (own_chat, owner)]:
            connection.execute(
                tables["chats"]
                .insert()
                .values(
                    id=chat_id,
                    title="preserve this conversation",
                    engine_id=1,
                    engine_options={},
                    user_id=user,
                    browser_id="legacy-browser",
                    visibility=1,
                )
            )
        for name, value in [
            ("langfuse_host", "https://cloud.langfuse.com"),
            ("langfuse_public_key", "pk-fixture"),
            ("langfuse_secret_key", "legacy-fixture"),
        ]:
            connection.execute(
                tables["site_settings"]
                .insert()
                .values(
                    name=name,
                    value=value,
                    data_type="str",
                )
            )
        connection.execute(
            tables["staff_action_logs"]
            .insert()
            .values(
                id=1,
                action="update",
                target_type="llm",
                target_id=1,
                before={},
                after={},
            )
        )

    # Follow the documented order: migrate provider ciphertext before schema
    # changes, with all writers assumed stopped. Dry-run must leave bytes intact.
    with database.begin() as connection:
        assert migrate_connection(
            connection, current_key, legacy_secret=legacy_key
        ) == {
            "checked": 3,
            "needs_migration": 3,
            "migrated": 0,
        }
        for name in providers:
            assert (
                connection.execute(text(f"SELECT credentials FROM {name}")).scalar_one()
                == original[name]
            )
        assert (
            migrate_connection(
                connection, current_key, legacy_secret=legacy_key, apply=True
            )["migrated"]
            == 3
        )
    with Session(database) as session:
        assert migrate_session(session, current_key)["needs_migration"] == 1
        assert migrate_session(session, current_key, apply=True)["migrated"] == 1
        session.commit()

    migrate(database, "upgrade", "head")
    with database.connect() as connection:
        assert (
            connection.execute(text("SELECT auth_version FROM users")).scalar_one() == 0
        )
        assert not connection.execute(
            text("SELECT is_active FROM api_keys")
        ).scalar_one()
        assert (
            connection.execute(text("SELECT COUNT(*) FROM user_sessions")).scalar_one()
            == 1
        )
        visibility = dict(
            connection.execute(text("SELECT id, visibility FROM chats")).all()
        )
        assert visibility == {guest_chat: 0, own_chat: 1}
        assert (
            connection.execute(
                text("SELECT actor_id FROM staff_action_logs")
            ).scalar_one()
            is None
        )
        for name in providers:
            encrypted = connection.execute(
                text(f"SELECT credentials FROM {name}")
            ).scalar_one()
            assert encrypted.startswith(MAGIC)
            assert json.loads(
                AESCipher(current_key, context=f"{name}.credentials").decrypt(encrypted)
            ) == {
                "api_key": "fixture-" + name,
            }
    with Session(database) as session:
        assert read_runtime_configuration(session).secret_key == "legacy-fixture"

    # Schema rollback cannot reactivate unrestricted keys or republish guests.
    # Credential ciphertext still requires the documented matching data/key
    # restore before running an old application; downgrade does not decrypt it.
    migrate(database, "downgrade", "04947f9684ab")
    with database.connect() as connection:
        assert not connection.execute(
            text("SELECT is_active FROM api_keys")
        ).scalar_one()
        assert (
            connection.execute(text("SELECT COUNT(*) FROM user_sessions")).scalar_one()
            == 0
        )
        visibility = dict(
            connection.execute(text("SELECT id, visibility FROM chats")).all()
        )
        assert visibility == {guest_chat: 0, own_chat: 1}
    migrate(database, "upgrade", "head")
    with database.begin() as connection:
        assert migrate_connection(connection, current_key) == {
            "checked": 3,
            "needs_migration": 0,
            "migrated": 0,
        }
