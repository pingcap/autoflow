import logging

from fastapi import APIRouter, Depends
from app.api.deps import SessionDep
from fastapi_pagination import Params, Page

from app.models.chat_engine import ChatEngine
from app.rag.chat.config import ChatEngineConfig
from app.repositories.chat_engine import chat_engine_repo

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/chat-engines")
def list_chat_engines(
    db_session: SessionDep,
    params: Params = Depends(),
) -> Page[ChatEngine]:
    page = chat_engine_repo.paginate(db_session, params, need_public=True)
    public_items = []
    for chat_engine in page.items:
        engine_config = ChatEngineConfig.model_validate(chat_engine.engine_options)
        public_items.append(
            chat_engine.model_copy(
                update={"engine_options": engine_config.screenshot()}
            )
        )
    return page.model_copy(update={"items": public_items})


@router.get("/chat-engines/{chat_engine_id}")
def get_chat_engine(
    db_session: SessionDep,
    chat_engine_id: int,
) -> ChatEngine:
    chat_engine = chat_engine_repo.must_get(
        db_session, chat_engine_id, need_public=True
    )
    engine_config = ChatEngineConfig.model_validate(chat_engine.engine_options)
    return chat_engine.model_copy(update={"engine_options": engine_config.screenshot()})
