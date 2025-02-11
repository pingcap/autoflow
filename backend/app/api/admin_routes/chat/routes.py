from typing import Optional
from fastapi import APIRouter, Depends
from fastapi_pagination import Page, Params

from app.models.chat import ChatOrigin
from app.api.deps import CurrentSuperuserDep, SessionDep
from app.repositories import chat_repo


router = APIRouter(
    prefix="/admin/chats",
    tags=["admin/chats"],
)


@router.get("/origins")
def list_chat_origins(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    search: Optional[str] = None,
    params: Params = Depends(),
) -> Page[ChatOrigin]:
    return chat_repo.list_chat_origins(db_session, search, params)
