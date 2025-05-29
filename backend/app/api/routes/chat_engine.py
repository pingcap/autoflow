import logging

from fastapi import APIRouter, Depends
from app.api.deps import SessionDep
from fastapi_pagination import Params, Page

from app.models.chat_engine import ChatEngine
from app.repositories.chat_engine import chat_engine_repo

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/chat-engines")
def list_chat_engines(
    db_session: SessionDep,
    params: Params = Depends(),
) -> Page[ChatEngine]:
    page = chat_engine_repo.paginate(db_session, params, need_public=True)
    for item in page.items:
        if "post_verification_token" in item.engine_options:
            item.engine_options["post_verification_token"] = "********"
        if "post_verification_url" in item.engine_options:
            item.engine_options["post_verification_url"] = "********"
        if "external_engine_config" in item.engine_options:
            if "stream_chat_api_url" in item.engine_options["external_engine_config"]:
                item.engine_options["external_engine_config"]["stream_chat_api_url"] = (
                    "********"
                )
    return page


@router.get("/chat-engines/{chat_engine_id}")
def get_chat_engine(
    db_session: SessionDep,
    chat_engine_id: int,
) -> ChatEngine:
    chat_engine = chat_engine_repo.must_get(
        db_session, chat_engine_id, need_public=True
    )
    if "post_verification_token" in chat_engine.engine_options:
        chat_engine.engine_options["post_verification_token"] = "********"
    if "post_verification_url" in chat_engine.engine_options:
        chat_engine.engine_options["post_verification_url"] = "********"
    if "external_engine_config" in chat_engine.engine_options:
        if (
            "stream_chat_api_url"
            in chat_engine.engine_options["external_engine_config"]
        ):
            chat_engine.engine_options["external_engine_config"][
                "stream_chat_api_url"
            ] = "********"
    return chat_engine
