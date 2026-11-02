import string
import secrets
import hashlib
import re
from datetime import datetime, timedelta, UTC
from pydantic import ValidationError
from fastapi import HTTPException
from app.auth.key_policy import ApiKeyGrant
from app.core.config import settings
from typing import Optional, Tuple

from fastapi import Request
from sqlmodel import Session, select
from sqlmodel.ext.asyncio.session import AsyncSession
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import apaginate

from app.models import ApiKey, User, KnowledgeBase


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


def hash_api_key(api_key: str) -> str:
    # Keys contain 50 cryptographically random alphanumeric characters. A fast
    # domain-separated digest is sufficient for these high-entropy credentials.
    # The v2 format also prevents an old worker that ignores scopes from accepting
    # newly issued credentials during a deployment.
    return (
        "sha256-v2$"
        + hashlib.sha256(b"autoflow-api-key-v2\0" + api_key.encode()).hexdigest()
    )


class ApiKeyManager:
    async def acreate_api_key(
        self, session: AsyncSession, user: User, description: str, grant: ApiKeyGrant
    ) -> Tuple[ApiKey, str]:
        api_key = generate_api_key()
        hashed_api_key = hash_api_key(api_key)
        api_key_obj = ApiKey(
            hashed_secret=hashed_api_key,
            api_key_display=api_key[:7] + "...." + api_key[-3:],
            user_id=user.id,
            auth_version=user.auth_version,
            description=description,
            scopes=sorted(set(grant.scopes)),
            knowledge_base_ids=sorted(set(grant.knowledge_base_ids)),
            expires_at=datetime.now(UTC) + timedelta(days=grant.expires_in_days),
        )
        session.add(api_key_obj)
        await session.commit()
        await session.refresh(api_key_obj)
        return api_key_obj, api_key

    def create_api_key(
        self, session: Session, user: User, description: str, grant: ApiKeyGrant
    ) -> Tuple[ApiKey, str]:
        api_key = generate_api_key()
        hashed_api_key = hash_api_key(api_key)
        api_key_obj = ApiKey(
            hashed_secret=hashed_api_key,
            api_key_display=api_key[:7] + "...." + api_key[-3:],
            user_id=user.id,
            auth_version=user.auth_version,
            description=description,
            scopes=sorted(set(grant.scopes)),
            knowledge_base_ids=sorted(set(grant.knowledge_base_ids)),
            expires_at=datetime.now(UTC) + timedelta(days=grant.expires_in_days),
        )
        session.add(api_key_obj)
        session.commit()
        session.refresh(api_key_obj)
        return api_key_obj, api_key

    async def _authenticate(self, session: AsyncSession, api_key: str | None):
        if not api_key or not re.fullmatch(r"ta-[A-Za-z0-9]{50}", api_key):
            return None
        hashed_api_key = hash_api_key(api_key)
        results = await session.exec(
            select(ApiKey).where(
                ApiKey.is_active == True,
                ApiKey.hashed_secret == hashed_api_key,
                ApiKey.expires_at > datetime.now(UTC),
            )
        )
        key = results.first()
        if not key or not key.expires_at:
            return None
        expiry = (
            key.expires_at.replace(tzinfo=UTC)
            if key.expires_at.tzinfo is None
            else key.expires_at
        )
        if expiry <= datetime.now(UTC):
            return None
        try:
            ApiKeyGrant(scopes=key.scopes, knowledge_base_ids=key.knowledge_base_ids)
        except ValidationError:
            return None
        user = await session.get(User, key.user_id)
        if (
            not user
            or key.auth_version != user.auth_version
            or not (user.is_active and user.is_verified)
        ):
            return None
        return key, user

    async def get_active_user_by_raw_api_key(
        self, session: AsyncSession, api_key: str
    ) -> Optional[User]:
        authenticated = await self._authenticate(session, api_key)
        return authenticated[1] if authenticated else None

    async def get_active_user_from_request(
        self, session: AsyncSession, request: Request
    ) -> Optional[User]:
        raw_key = get_api_key_from_request(request)
        if raw_key is None:
            return None
        authenticated = await self._authenticate(session, raw_key)
        if not authenticated:
            raise HTTPException(status_code=401, detail="Invalid API key")
        key, user = authenticated
        required_scope = {
            f"{settings.API_V1_STR}/retrieve/chunks": "retrieve:chunks",
            f"{settings.API_V1_STR}/retrieve/knowledge_graph": "retrieve:graph",
        }.get(request.url.path)
        if (
            request.method != "POST"
            or not required_scope
            or required_scope not in key.scopes
        ):
            raise HTTPException(
                status_code=403, detail="API key does not permit this operation"
            )
        try:
            payload = await request.json()
            ids = payload["retrieval_config"]["knowledge_base_ids"]
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 100
                or any(type(item) is not int or item <= 0 for item in ids)
            ):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise HTTPException(
                status_code=403, detail="An explicit knowledge-base scope is required"
            ) from None
        if not set(ids).issubset(key.knowledge_base_ids):
            raise HTTPException(
                status_code=403, detail="Knowledge base is outside this API key's scope"
            )
        available = await session.exec(
            select(KnowledgeBase.id).where(
                KnowledgeBase.id.in_(ids), KnowledgeBase.deleted_at == None
            )
        )
        if set(available.all()) != set(ids):
            raise HTTPException(status_code=403, detail="Knowledge base is unavailable")
        request.state.auth_method = "api_key"
        request.state.api_key_id = key.id
        request.state.user_id = str(user.id)
        return user

    async def list_api_keys(
        self, session: AsyncSession, user: User, params: Params
    ) -> Page[ApiKey]:
        api_keys = await apaginate(
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
