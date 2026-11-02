import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


VERSIONS = Path(__file__).parents[2] / "app" / "alembic" / "versions"


def migration(filename, connection):
    spec = importlib.util.spec_from_file_location(filename, VERSIONS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.op = Operations(MigrationContext.configure(connection))
    return module


def test_upgrade_and_rollback_never_reactivate_old_credentials_or_guest_links():
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        connection.execute(text("CREATE TABLE user_sessions (token TEXT PRIMARY KEY)"))
        connection.execute(
            text(
                "CREATE TABLE api_keys (id INTEGER PRIMARY KEY, is_active BOOLEAN NOT NULL)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE chats (id INTEGER PRIMARY KEY, user_id INTEGER, visibility INTEGER)"
            )
        )
        connection.execute(text("INSERT INTO users VALUES (1)"))
        connection.execute(text("INSERT INTO user_sessions VALUES ('old-session')"))
        connection.execute(text("INSERT INTO api_keys VALUES (1, true)"))
        connection.execute(
            text("INSERT INTO chats VALUES (1, NULL, 1), (2, 1, 1), (3, NULL, 0)")
        )
        versions = [
            migration(filename, connection)
            for filename in [
                "4b8e2a70c1d9_credential_versions.py",
                "65ad0f3c2b81_scoped_expiring_api_keys.py",
                "8f4c62e10a93_private_legacy_guest_chats.py",
                "0d7c24a5b6f8_scoped_chat_keys.py",
            ]
        ]
        for version in versions:
            version.upgrade()
        assert (
            connection.execute(text("SELECT auth_version FROM users")).scalar_one() == 0
        )
        assert (
            connection.execute(text("SELECT is_active FROM api_keys")).scalar_one() == 0
        )
        assert connection.execute(
            text("SELECT visibility FROM chats ORDER BY id")
        ).scalars().all() == [0, 1, 0]
        assert {"scopes", "knowledge_base_ids", "expires_at"}.issubset(
            {column["name"] for column in inspect(connection).get_columns("api_keys")}
        )
        connection.execute(
            text(
                "INSERT INTO api_keys (id, is_active, auth_version) VALUES (2, true, 1)"
            )
        )
        for version in reversed(versions):
            version.downgrade()
        assert (
            connection.execute(text("SELECT count(*) FROM user_sessions")).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text("SELECT count(*) FROM api_keys WHERE is_active = true")
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text("SELECT visibility FROM chats WHERE id = 1")
            ).scalar_one()
            == 0
        )
    engine.dispose()
