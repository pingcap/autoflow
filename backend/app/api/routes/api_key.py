from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from app.models import KnowledgeBase
from app.auth.key_policy import ApiKeyGrant
from pydantic import BaseModel, Field
from fastapi_pagination import Params, Page

from app.api.deps import AsyncSessionDep, CurrentSuperuserDep
from app.auth.api_keys import api_key_manager
from app.models import PublicApiKey

router = APIRouter()


class CreateApiKeyRequest(ApiKeyGrant):
    description: str = Field(min_length=1, max_length=100)


class CreateApiKeyResponse(BaseModel):
    api_key: str


@router.post("/api-keys")
async def create_api_key(
    session: AsyncSessionDep, user: CurrentSuperuserDep, request: CreateApiKeyRequest
) -> CreateApiKeyResponse:
    result = await session.exec(
        select(KnowledgeBase.id).where(
            KnowledgeBase.id.in_(request.knowledge_base_ids),
            KnowledgeBase.deleted_at == None,
        )
    )
    if set(result.all()) != set(request.knowledge_base_ids):
        raise HTTPException(status_code=400, detail="Select existing knowledge bases")
    grant = ApiKeyGrant.model_validate(request.model_dump(exclude={"description"}))
    _, raw_api_key = await api_key_manager.acreate_api_key(
        session, user, request.description, grant
    )
    return CreateApiKeyResponse(api_key=raw_api_key)


@router.get("/api-keys")
async def list_api_keys(
    session: AsyncSessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> Page[PublicApiKey]:
    return await api_key_manager.list_api_keys(session, user, params)


@router.delete("/api-keys/{api_key_id}")
async def delete_api_key(
    session: AsyncSessionDep, user: CurrentSuperuserDep, api_key_id: int
):
    return await api_key_manager.delete_api_key(session, user, api_key_id)
