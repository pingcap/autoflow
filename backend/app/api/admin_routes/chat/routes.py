from fastapi import APIRouter, Depends
from fastapi_pagination import Params

from app.models.chat import ChatOrigin
from app.api.deps import CurrentSuperuserDep, SessionDep
from app.repositories import chat_repo


router = APIRouter(
    prefix="/admin/chats",
    tags=["admin/chat"],
)


@router.get("/origins")
def list_chat_origins(
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> list[ChatOrigin]:
    return chat_repo.list_chat_origins(session, params)
