from fastapi import Depends
from fastapi_users_db_sqlmodel import SQLModelUserDatabaseAsync
from fastapi_users_db_sqlmodel.access_token import SQLModelAccessTokenDatabaseAsync
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import User, UserSession, ApiKey
from sqlmodel import delete, update
from app.core.db import get_db_async_session
from app.security.audit import audit_context


class RevokingUserDatabase(SQLModelUserDatabaseAsync):
    async def update(self, user: User, update_dict: dict) -> User:
        context = audit_context.get()
        if context and context.token_flow in {"recovery", "verification"}:
            # FastAPI Users reaches this database mutation only after validating
            # the recovery/verification token and its subject. Do not attribute
            # it to an unrelated session cookie on the same browser request.
            context.actor_id = str(user.id)
            context.actor_type = context.token_flow
            context.api_key_id = None
            context.privileged = False
        credential_fields = {
            "hashed_password",
            "email",
            "is_active",
            "is_verified",
            "is_superuser",
        }
        if not credential_fields.intersection(update_dict):
            return await super().update(user, update_dict)
        try:
            # Increment in SQL so concurrent credential changes cannot reuse an
            # epoch. A token minted by an already-running login retains its old
            # epoch and is rejected even if inserted after this deletion.
            await self.session.exec(
                update(User)
                .where(User.id == user.id)
                .values(**update_dict, auth_version=User.auth_version + 1)
                .execution_options(synchronize_session=False)
            )
            await self.session.exec(
                delete(UserSession).where(UserSession.user_id == user.id)
            )
            await self.session.exec(
                update(ApiKey).where(ApiKey.user_id == user.id).values(is_active=False)
            )
            await self.session.commit()
            await self.session.refresh(user)
            return user
        except Exception:
            await self.session.rollback()
            raise


async def get_user_db(session: AsyncSession = Depends(get_db_async_session)):
    yield RevokingUserDatabase(session, User)


async def get_user_session_db(
    session: AsyncSession = Depends(get_db_async_session),
):
    yield SQLModelAccessTokenDatabaseAsync(session, UserSession)
