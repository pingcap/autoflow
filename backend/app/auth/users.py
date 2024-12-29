import logging
import uuid
import contextlib
from http import HTTPStatus
from typing import Optional

from fastapi import Depends, Request, HTTPException
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import (
    AuthenticationBackend,
    CookieTransport,
)
from fastapi_users.authentication.strategy import DatabaseStrategy
from fastapi_users_db_sqlmodel import SQLModelUserDatabaseAsync
from fastapi_users_db_sqlmodel.access_token import SQLModelAccessTokenDatabaseAsync
from fastapi_users.exceptions import UserAlreadyExists, UserNotExists
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings, Environment
from app.core.db import get_db_async_session
from app.models import User, UserSession
from app.auth.db import get_user_db, get_user_session_db
from app.auth.api_keys import api_key_manager
from app.auth.schemas import UserCreate, UserUpdate

logger = logging.getLogger(__name__)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = settings.SECRET_KEY
    verification_token_secret = settings.SECRET_KEY

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        print(f"User {user.id} has registered.")

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        print(f"User {user.id} has forgot their password. Reset token: {token}")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        print(f"Verification requested for user {user.id}. Verification token: {token}")


async def get_user_manager(user_db: SQLModelUserDatabaseAsync = Depends(get_user_db)):
    yield UserManager(user_db)


cookie_transport = CookieTransport(
    cookie_name=settings.SESSION_COOKIE_NAME,
    cookie_max_age=settings.SESSION_COOKIE_MAX_AGE,
    cookie_secure=settings.SESSION_COOKIE_SECURE,
)


def get_database_strategy(
    user_session_db: SQLModelAccessTokenDatabaseAsync[UserSession] = Depends(
        get_user_session_db
    ),
) -> DatabaseStrategy:
    return DatabaseStrategy(user_session_db, lifetime_seconds=3600 * 24 * 90)


auth_backend = AuthenticationBackend(
    name="database",
    transport=cookie_transport,
    get_strategy=get_database_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])


# Following methods are used to get the current user from the request,
# They all support both session cookies and API keys for authentication,
# it will first check for a session cookie, if not found, then check for an API key.
async def current_user(
    request: Request,
    user: User = Depends(
        fastapi_users.current_user(optional=True, active=True, verified=True)
    ),
    session: AsyncSession = Depends(get_db_async_session),
) -> User:
    if user:
        # already authenticated with a valid session cookie
        return user

    # check for an API key
    user = await api_key_manager.get_active_user_from_request(session, request)
    if not user:
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED)
    return user


async def current_superuser(
    request: Request,
    user: User = Depends(
        fastapi_users.current_user(optional=True, active=True, verified=True)
    ),
    session: AsyncSession = Depends(get_db_async_session),
) -> User:
    if user:
        if user.is_superuser:
            return user
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN)

    # check for an API key
    user = await api_key_manager.get_active_user_from_request(session, request)
    if not user:
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED)
    if not user.is_superuser:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN)
    return user


async def optional_current_user(
    request: Request,
    user: User = Depends(
        fastapi_users.current_user(optional=True, active=True, verified=True)
    ),
    session: AsyncSession = Depends(get_db_async_session),
) -> Optional[User]:
    if user:
        # already authenticated with a valid session cookie
        return user

    # check for an API key
    return await api_key_manager.get_active_user_from_request(session, request)


get_user_db_context = contextlib.asynccontextmanager(get_user_db)
get_user_manager_context = contextlib.asynccontextmanager(get_user_manager)


async def create_user(
    session: AsyncSession,
    email: str,
    password: str,
    is_active: bool = True,
    is_verified: bool = True,
    is_superuser: bool = False,
) -> User:
    try:
        async with get_user_db_context(session) as user_db:
            async with get_user_manager_context(user_db) as user_manager:
                user = await user_manager.create(
                    UserCreate(
                        email=email,
                        password=password,
                        is_active=is_active,
                        is_verified=is_verified,
                        is_superuser=is_superuser,
                    )
                )
                return user
    except UserAlreadyExists:
        logger.error(f"User {email} already exists")
        raise


async def update_user_password(
    session: AsyncSession,
    user_id: uuid.UUID,
    new_password: str,
) -> User:
    try:
        async with get_user_db_context(session) as user_db:
            async with get_user_manager_context(user_db) as user_manager:
                user = await user_manager.get(user_id)
                if not user:
                    raise UserNotExists(f"User {user_id} does not exist")

                user_update = UserUpdate(password=new_password)
                await user_manager.update(user_update, user)
                # verify
                updated_user = await user_manager.get(user_id)
                return updated_user

    except UserNotExists as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Failed to update password for user {id}: {e}")
        raise
