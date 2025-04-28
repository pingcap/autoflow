import base64
import string
import secrets
import hashlib
from typing import Optional, Tuple

from fastapi import Request
from sqlmodel import Session, select
from sqlmodel.ext.asyncio.session import AsyncSession
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate

from app.models import ApiKey, User


def generate_api_key(length=50):
    characters = string.ascii_letters + string.digits
    api_key = "".join(secrets.choice(characters) for _ in range(length))
    return "ta-" + api_key


API_KEY_HEADER = "Authorization"
BEARER_PREFIX = "Bearer "


def get_api_key_from_request(request: Request) -> str | None:
    api_key_header = request.headers.get(API_KEY_HEADER)
    if api_key_header is None:
        return None

    if not api_key_header.startswith(BEARER_PREFIX):
        return None

    return api_key_header[len(BEARER_PREFIX) :].strip()


def encrypt_api_key(api_key: str) -> str:
    # An empty salt is used because we need to look tokens up solely by
    # their hashed value. Additionally, tokens are always cryptographically
    # pseudo-random and unique, therefore salting provides no
    # additional security.
    algorithm = "pbkdf2_sha512"
    api_key = api_key.encode("utf-8")
    salt = b""
    iterations = 20_000
    hash = hashlib.pbkdf2_hmac("sha512", api_key, salt, iterations)
    hash = base64.b64encode(hash).decode("ascii").strip()
    return "%s$%d$%s" % (algorithm, iterations, hash)


class ApiKeyManager:
    async def acreate_api_key(
        self, session: AsyncSession, user: User, description: str
    ) -> Tuple[ApiKey, str]:
        api_key = generate_api_key()
        hashed_api_key = encrypt_api_key(api_key)
        api_key_obj = ApiKey(
            hashed_secret=hashed_api_key,
            api_key_display=api_key[:7] + "...." + api_key[-3:],
            user_id=user.id,
            description=description,
        )
        session.add(api_key_obj)
        await session.commit()
        await session.refresh(api_key_obj)
        return api_key_obj, api_key

    def create_api_key(
        self, session: Session, user: User, description: str
    ) -> Tuple[ApiKey, str]:
        api_key = generate_api_key()
        hashed_api_key = encrypt_api_key(api_key)
        api_key_obj = ApiKey(
            hashed_secret=hashed_api_key,
            api_key_display=api_key[:7] + "...." + api_key[-3:],
            user_id=user.id,
            description=description,
        )
        session.add(api_key_obj)
        session.commit()
        session.refresh(api_key_obj)
        return api_key_obj, api_key

    async def get_active_user_by_raw_api_key(
        self, session: AsyncSession, api_key: str
    ) -> Optional[User]:
        if not api_key:
            return None
        hashed_api_key = encrypt_api_key(api_key)
        results = await session.exec(
            select(ApiKey).where(
                ApiKey.is_active == True,
                ApiKey.hashed_secret == hashed_api_key,
            )
        )
        api_key_obj = results.first()
        if not api_key_obj:
            return None

        user = await session.get(User, api_key_obj.user_id)
        if not (user.is_active and user.is_verified):
            return None
        return user

    async def get_active_user_from_request(
        self, session: AsyncSession, request: Request
    ) -> Optional[User]:
        api_key = get_api_key_from_request(request)
        return await self.get_active_user_by_raw_api_key(session, api_key)

    async def list_api_keys(
        self, session: AsyncSession, user: User, params: Params
    ) -> Page[ApiKey]:
        api_keys = await paginate(
            session,
            select(ApiKey)
            .where(ApiKey.user == user, ApiKey.is_active == True)
            .order_by(ApiKey.created_at.desc()),
            params,
        )
        return api_keys

    async def delete_api_key(self, session: AsyncSession, user: User, api_key_id: int):
        result = await session.exec(
            select(ApiKey).where(
                ApiKey.id == api_key_id,
                ApiKey.user_id == user.id,
                ApiKey.is_active == True,
            )
        )
        api_key = result.first()
        if api_key:
            api_key.is_active = False
            await session.commit()


api_key_manager = ApiKeyManager()
