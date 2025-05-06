from fastapi import APIRouter, Depends
from fastapi_pagination import Params, Page

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.exceptions import DefaultChatEngineCannotBeDeleted
from app.rag.chat.config import ChatEngineConfig
from app.repositories import chat_engine_repo
from app.models import ChatEngine, ChatEngineUpdate

router = APIRouter()


@router.get("/admin/chat-engines")
def list_chat_engines(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> Page[ChatEngine]:
    return chat_engine_repo.paginate(db_session, params)


@router.post("/admin/chat-engines")
def create_chat_engine(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    chat_engine: ChatEngine,
) -> ChatEngine:
    return chat_engine_repo.create(db_session, chat_engine)


@router.get("/admin/chat-engines/{chat_engine_id}")
def get_chat_engine(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    chat_engine_id: int,
) -> ChatEngine:
    return chat_engine_repo.must_get(db_session, chat_engine_id)


@router.put("/admin/chat-engines/{chat_engine_id}")
def update_chat_engine(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    chat_engine_id: int,
    update: ChatEngineUpdate,
) -> ChatEngine:
    chat_engine = chat_engine_repo.must_get(db_session, chat_engine_id)
    return chat_engine_repo.update(db_session, chat_engine, update)


@router.delete("/admin/chat-engines/{chat_engine_id}")
def delete_chat_engine(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    chat_engine_id: int,
) -> ChatEngine:
    chat_engine = chat_engine_repo.must_get(db_session, chat_engine_id)
    if chat_engine.is_default:
        raise DefaultChatEngineCannotBeDeleted(chat_engine_id)
    return chat_engine_repo.delete(db_session, chat_engine)


@router.get("/admin/chat-engines-default-config")
def get_default_config(
    db_session: SessionDep, user: CurrentSuperuserDep
) -> ChatEngineConfig:
    return ChatEngineConfig()
