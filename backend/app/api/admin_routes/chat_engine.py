from fastapi import APIRouter, Depends, HTTPException, status
from fastapi_pagination import Params, Page

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.chat_config import ChatEngineConfig
from app.repositories import chat_engine_repo
from app.models import ChatEngine, ChatEngineUpdate

router = APIRouter()


@router.get("/admin/chat-engines")
def list_chat_engines(
    session: SessionDep, user: CurrentSuperuserDep, params: Params = Depends()
) -> Page[ChatEngine]:
    return chat_engine_repo.paginate(session, params)


@router.post("/admin/chat-engines")
def create_chat_engine(
    chat_engine: ChatEngine, session: SessionDep, user: CurrentSuperuserDep
) -> ChatEngine:
    return chat_engine_repo.create(session, chat_engine)


@router.get("/admin/chat-engines/{chat_engine_id}")
def get_chat_engine(
    chat_engine_id: int, session: SessionDep, user: CurrentSuperuserDep
) -> ChatEngine:
    chat_engine = chat_engine_repo.get(session, chat_engine_id)
    if chat_engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat engine not found"
        )
    return chat_engine


@router.put("/admin/chat-engines/{chat_engine_id}")
def update_chat_engine(
    chat_engine_id: int,
    session: SessionDep,
    user: CurrentSuperuserDep,
    chat_engine_in: ChatEngineUpdate,
) -> ChatEngine:
    chat_engine = chat_engine_repo.get(session, chat_engine_id)
    if chat_engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat engine not found"
        )
    return chat_engine_repo.update(session, chat_engine, chat_engine_in)


@router.delete("/admin/chat-engines/{chat_engine_id}")
def delete_chat_engine(
    chat_engine_id: int, session: SessionDep, user: CurrentSuperuserDep
) -> ChatEngine:
    chat_engine = chat_engine_repo.get(session, chat_engine_id)
    if chat_engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat engine not found"
        )
    if chat_engine.is_default:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the default chat engine",
        )
    return chat_engine_repo.delete(session, chat_engine)


@router.get("/admin/chat-engines-default-config")
def get_default_config(
    session: SessionDep, user: CurrentSuperuserDep
) -> ChatEngineConfig:
    return ChatEngineConfig()
