from fastapi import APIRouter, Depends
from pydantic import BaseModel
from fastapi_pagination import Params, Page

from app.api.deps import AsyncSessionDep, CurrentSuperuserDep
from app.auth.api_keys import api_key_manager
from app.models import PublicApiKey

router = APIRouter()


class CreateApiKeyRequest(BaseModel):
    description: str


class CreateApiKeyResponse(BaseModel):
    api_key: str


@router.post("/api-keys")
async def create_api_key(
    session: AsyncSessionDep, user: CurrentSuperuserDep, request: CreateApiKeyRequest
) -> CreateApiKeyResponse:
    _, raw_api_key = await api_key_manager.acreate_api_key(
        session, user, request.description
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
