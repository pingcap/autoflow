from app.auth.chat_access import user_can_use_engine
from app.utils.public_chat import chat_hides_sources, public_chat_message
import logging
from uuid import UUID
from typing import List, Optional, Annotated, Literal
from http import HTTPStatus

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    Field,
)
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from fastapi_pagination import Params, Page
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from app.api.deps import SessionDep, OptionalUserDep
from app.rag.chat.chat_flow import ChatFlow
from app.rag.retrievers.knowledge_graph.schema import KnowledgeGraphRetrievalResult
from app.repositories import chat_repo
from app.models import Chat, ChatUpdate

from app.rag.chat.chat_service import get_final_chat_result
from app.models import Chat, ChatUpdate, ChatFilters
from app.rag.chat.chat_service import (
    user_can_view_chat,
    user_can_edit_chat,
    get_chat_message_subgraph,
    get_chat_message_recommend_questions,
    remove_chat_message_recommend_questions,
)
from app.exceptions import InternalServerError

logger = logging.getLogger(__name__)

router = APIRouter()


class ClientChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    messages: List[ClientChatMessage] = Field(min_length=1, max_length=20)
    chat_engine: str = Field(default="default", min_length=1, max_length=100)
    chat_id: Optional[UUID] = None
    stream: bool = True

    @field_validator("messages")
    @classmethod
    def check_messages(
        cls, messages: List[ClientChatMessage]
    ) -> List[ClientChatMessage]:
        if not messages:
            raise ValueError("messages cannot be empty")
        for m in messages:
            if m.role not in [MessageRole.USER, MessageRole.ASSISTANT]:
                raise ValueError("role must be either 'user' or 'assistant'")
            if not m.content:
                raise ValueError("message content cannot be empty")
            if len(m.content) > 20000:
                raise ValueError("message content cannot exceed 20000 characters")
        if sum(len(m.content) for m in messages) > 40000:
            raise ValueError("total message content cannot exceed 40000 characters")
        if messages[-1].role != MessageRole.USER:
            raise ValueError("last message must be from user")
        return messages


@router.post("/chats")
def chats(
    request: Request,
    session: SessionDep,
    user: OptionalUserDep,
    chat_request: ChatRequest,
):
    origin = request.headers.get("Origin") or request.headers.get("Referer")
    browser_id = request.state.browser_id

    try:
        chat_flow = ChatFlow(
            db_session=session,
            user=user,
            browser_id=browser_id,
            origin=origin,
            chat_id=chat_request.chat_id,
            chat_messages=[
                ChatMessage(role=m.role, content=m.content)
                for m in chat_request.messages
            ],
            engine_name=getattr(
                request.state, "api_key_engine_name", chat_request.chat_engine
            ),
            api_key_id=getattr(request.state, "api_key_id", None),
            allowed_engine_ids=getattr(request.state, "api_key_engine_ids", None),
        )

        if chat_request.stream:
            return StreamingResponse(
                chat_flow.chat(),
                media_type="text/event-stream",
                headers={
                    "X-Content-Type-Options": "nosniff",
                },
            )
        else:
            return get_final_chat_result(chat_flow.chat())
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/chats")
def list_chats(
    request: Request,
    session: SessionDep,
    user: OptionalUserDep,
    filters: Annotated[ChatFilters, Query()],
    params: Params = Depends(),
) -> Page[Chat]:
    browser_id = request.state.browser_id
    return chat_repo.paginate(session, user, browser_id, filters, params)


@router.get("/chats/{chat_id}")
def get_chat(
    request: Request, session: SessionDep, user: OptionalUserDep, chat_id: UUID
):
    chat = chat_repo.must_get(session, chat_id)

    if not user_can_view_chat(chat, user, request.state.browser_id):
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="Access denied")

    return {
        "chat": chat,
        "messages": [
            public_chat_message(message, chat_hides_sources(chat))
            for message in chat_repo.get_messages(session, chat)
        ],
    }


@router.put("/chats/{chat_id}")
def update_chat(
    request: Request,
    session: SessionDep,
    user: OptionalUserDep,
    chat_id: UUID,
    chat_update: ChatUpdate,
):
    try:
        chat = chat_repo.must_get(session, chat_id)

        if not user_can_edit_chat(chat, user, request.state.browser_id):
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN, detail="Access denied"
            )

        return chat_repo.update(session, chat, chat_update)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e, exc_info=True)
        raise InternalServerError()


@router.delete("/chats/{chat_id}")
def delete_chat(
    request: Request, session: SessionDep, user: OptionalUserDep, chat_id: UUID
):
    try:
        chat = chat_repo.must_get(session, chat_id)

        if not user_can_edit_chat(chat, user, request.state.browser_id):
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN, detail="Access denied"
            )

        return chat_repo.delete(session, chat)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e, exc_info=True)
        raise InternalServerError()


@router.get(
    "/chat-messages/{chat_message_id}/subgraph",
    response_model=KnowledgeGraphRetrievalResult,
)
def get_chat_subgraph(
    request: Request, session: SessionDep, user: OptionalUserDep, chat_message_id: int
):
    try:
        chat_message = chat_repo.must_get_message(session, chat_message_id)

        if not user_can_view_chat(chat_message.chat, user, request.state.browser_id):
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN, detail="Access denied"
            )

        if not user_can_use_engine(chat_message.chat.engine, user):
            raise HTTPException(status_code=404, detail="Chat engine not found")
        if chat_hides_sources(chat_message.chat):
            raise HTTPException(status_code=404, detail="Sources are hidden")
        result = get_chat_message_subgraph(session, chat_message)
        return result.model_dump(exclude_none=True)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e, exc_info=True)
        raise InternalServerError()


@router.get("/chat-messages/{chat_message_id}/recommended-questions")
def get_recommended_questions(
    request: Request, session: SessionDep, user: OptionalUserDep, chat_message_id: int
) -> List[str]:
    try:
        chat_message = chat_repo.must_get_message(session, chat_message_id)

        if not user_can_view_chat(chat_message.chat, user, request.state.browser_id):
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN, detail="Access denied"
            )

        if not user_can_use_engine(chat_message.chat.engine, user):
            raise HTTPException(status_code=404, detail="Chat engine not found")
        return get_chat_message_recommend_questions(
            session,
            chat_message,
            engine_name=chat_message.chat.engine.name,
            allow_generate=False,
            need_public=not (user and user.is_superuser),
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e, exc_info=True)
        raise InternalServerError()


@router.post("/chat-messages/{chat_message_id}/recommended-questions")
def refresh_recommended_questions(
    request: Request,
    session: SessionDep,
    user: OptionalUserDep,
    chat_message_id: int,
    refresh: bool = Query(default=True),
) -> List[str]:
    try:
        chat_message = chat_repo.must_get_message(session, chat_message_id)

        can_generate = user_can_edit_chat(
            chat_message.chat, user, request.state.browser_id
        )
        if not user_can_view_chat(
            chat_message.chat, user, request.state.browser_id
        ) or (refresh and not can_generate):
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN, detail="Access denied"
            )

        if not user_can_use_engine(chat_message.chat.engine, user):
            raise HTTPException(status_code=404, detail="Chat engine not found")
        if refresh:
            remove_chat_message_recommend_questions(session, chat_message_id)
        return get_chat_message_recommend_questions(
            session,
            chat_message,
            engine_name=chat_message.chat.engine.name,
            allow_generate=can_generate,
            need_public=not (user and user.is_superuser),
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e, exc_info=True)
        raise InternalServerError()
