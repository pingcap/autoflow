import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel import SQLModel, select
from fastapi_users_db_sqlmodel.access_token import SQLModelAccessTokenDatabaseAsync

from app.auth.db import RevokingUserDatabase
from app.auth.key_policy import ApiKeyGrant
from app.auth.api_keys import api_key_manager
from app.auth.session_strategy import VersionedDatabaseStrategy
from app.auth.users import UserManager
from app.models import User, UserSession, ApiKey


GRANT = ApiKeyGrant(scopes=["retrieve:chunks"], knowledge_base_ids=[1])


async def scenario(fail_revoke=False, *, database_url=None):
    engine = create_async_engine(database_url or "sqlite+aiosqlite://")
    if database_url is None:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda connection: SQLModel.metadata.create_all(
                    connection,
                    tables=[User.__table__, UserSession.__table__, ApiKey.__table__],
                )
            )
    # SQLModel's async session supplies exec, used by the real application adapter.
    from sqlmodel.ext.asyncio.session import AsyncSession

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        user = User(
            email="member@example.test", hashed_password="old-hash", is_verified=True
        )
        session.add(user)
        await session.commit()
        user_id = user.id
        before_reset = SimpleNamespace(id=user_id, auth_version=user.auth_version)
        database = RevokingUserDatabase(session, User)
        strategy = VersionedDatabaseStrategy(
            SQLModelAccessTokenDatabaseAsync(session, UserSession), 3600
        )
        manager = SimpleNamespace(parse_id=lambda value: value, get=database.get)
        old_token = await strategy.write_token(user)
        _, old_key = await api_key_manager.acreate_api_key(
            session, user, "before reset", GRANT
        )

        if fail_revoke:

            def reject_key_update(
                conn, cursor, statement, parameters, context, executemany
            ):
                if statement.startswith("UPDATE api_keys"):
                    raise RuntimeError("injected revocation failure")

            event.listen(engine.sync_engine, "before_cursor_execute", reject_key_update)
            with pytest.raises(RuntimeError, match="injected"):
                await database.update(user, {"hashed_password": "new-hash"})
            event.remove(engine.sync_engine, "before_cursor_execute", reject_key_update)
            user = await database.get(user_id)
            assert user.hashed_password == "old-hash"
            assert user.auth_version == 0
            assert await strategy.read_token(old_token, manager) is not None
            assert (
                await api_key_manager.get_active_user_by_raw_api_key(session, old_key)
                is not None
            )
        else:
            user = await database.update(user, {"hashed_password": "new-hash"})
            assert user.auth_version == 1
            assert await strategy.read_token(old_token, manager) is None
            assert (
                await api_key_manager.get_active_user_by_raw_api_key(session, old_key)
                is None
            )
            assert (await session.exec(select(UserSession))).all() == []
            assert all(
                not key.is_active for key in (await session.exec(select(ApiKey))).all()
            )
            # A login/key issuance authenticated before the reset but finishing
            # afterwards must not resurrect credentials from the old epoch.
            late_token = await strategy.write_token(before_reset)
            _, late_key = await api_key_manager.acreate_api_key(
                session, before_reset, "late old login", GRANT
            )
            assert await strategy.read_token(late_token, manager) is None
            assert (
                await api_key_manager.get_active_user_by_raw_api_key(session, late_key)
                is None
            )
            new_token = await strategy.write_token(user)
            _, new_key = await api_key_manager.acreate_api_key(
                session, user, "new login", GRANT
            )
            assert await strategy.read_token(new_token, manager) is not None
            assert (
                await api_key_manager.get_active_user_by_raw_api_key(session, new_key)
                is not None
            )
    await engine.dispose()


def test_reset_revokes_existing_and_late_issued_credentials():
    asyncio.run(scenario())


def test_revocation_failure_rolls_back_the_password_change():
    asyncio.run(scenario(fail_revoke=True))


def test_password_and_verification_hooks_never_log_tokens(caplog, capsys):
    manager = UserManager(AsyncMock())
    user = User(email="member@example.test", hashed_password="not-logged")
    asyncio.run(manager.on_after_forgot_password(user, "SECRET_RESET_TOKEN"))
    asyncio.run(manager.on_after_request_verify(user, "SECRET_VERIFY_TOKEN"))
    output = caplog.text + capsys.readouterr().out
    assert "SECRET_RESET_TOKEN" not in output
    assert "SECRET_VERIFY_TOKEN" not in output
