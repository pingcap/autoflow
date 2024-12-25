import logging
from uuid import UUID
from typing import List, Optional
from http import HTTPStatus

from pydantic import (
    BaseModel,
    field_validator,
)
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi_pagination import Params, Page

from app.api.deps import SessionDep, OptionalUserDep, CurrentUserDep
from app.rag.chat_config import ChatEngineConfig
from app.repositories import chat_repo
from app.models import Chat, ChatUpdate
from app.rag.chat import (
    ChatService,
    ChatEvent,
    user_can_view_chat,
    user_can_edit_chat,
    get_chat_message_subgraph,
    get_chat_message_recommend_questions,
    remove_chat_message_recommend_questions,
)
from app.rag.types import (
    MessageRole,
    ChatMessage,
    ChatEventType,
    ChatMessageSate,
)
from app.exceptions import (
    ChatNotFound,
    KBNotFound,
    LLMException,
    EmbeddingModelException,
    RerankerModelException,
    InternalServerError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    chat_engine: str = "default"
    chat_id: Optional[UUID] = None
    stream: bool = True

    @field_validator("messages")
    @classmethod
    def check_messages(cls, messages: List[ChatMessage]) -> List[ChatMessage]:
        if not messages:
            raise ValueError("messages cannot be empty")
        for m in messages:
            if m.role not in [MessageRole.USER, MessageRole.ASSISTANT]:
                raise ValueError("role must be either 'user' or 'assistant'")
            if not m.content:
                raise ValueError("message content cannot be empty")
            if len(m.content) > 10000:
                raise ValueError("message content cannot exceed 1000 characters")
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
        chat_svc = ChatService(
            db_session=session,
            user=user,
            browser_id=browser_id,
            origin=origin,
            chat_id=chat_request.chat_id,
            chat_messages=chat_request.messages,
            engine_name=chat_request.chat_engine,
        )
    except ChatNotFound as e:
        raise e
    except KBNotFound as e:
        raise e
    except LLMException as e:
        raise e
    except EmbeddingModelException as e:
        raise e
    except RerankerModelException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()

    if chat_request.stream:
        return StreamingResponse(
            chat_svc.chat(),
            media_type="text/event-stream",
            headers={
                "X-Content-Type-Options": "nosniff",
            },
        )
    else:
        trace, sources, content = None, [], ""
        chat_id, message_id = None, None
        for m in chat_svc.chat():
            if not isinstance(m, ChatEvent):
                continue
            if m.event_type == ChatEventType.MESSAGE_ANNOTATIONS_PART:
                if m.payload.state == ChatMessageSate.SOURCE_NODES:
                    sources = m.payload.context
                elif m.payload.state == ChatMessageSate.TRACE:
                    trace = m.payload.context
            elif m.event_type == ChatEventType.TEXT_PART:
                content += m.payload
            elif m.event_type == ChatEventType.DATA_PART:
                chat_id = m.payload.chat.id
                message_id = m.payload.assistant_message.id
            elif m.event_type == ChatEventType.ERROR_PART:
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail=m.payload,
                )
            else:
                pass
        return {
            "chat_id": chat_id,
            "message_id": message_id,
            "trace": trace,
            "sources": sources,
            "content": content,
        }


@router.get("/chats")
def list_chats(
    request: Request,
    session: SessionDep,
    user: OptionalUserDep,
    params: Params = Depends(),
) -> Page[Chat]:
    browser_id = request.state.browser_id
    return chat_repo.paginate(session, user, browser_id, params)


@router.get("/chats/{chat_id}")
def get_chat(session: SessionDep, user: OptionalUserDep, chat_id: UUID):
    chat = chat_repo.get(session, chat_id)
    if not chat:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Chat not found")

    if not user_can_view_chat(chat, user):
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="Access denied")

    return {
        "chat": chat,
        "messages": chat_repo.get_messages(session, chat),
    }


@router.put("/chats/{chat_id}")
def update_chat(
    session: SessionDep, user: CurrentUserDep, chat_id: UUID, chat_update: ChatUpdate
):
    chat = chat_repo.get(session, chat_id)
    if not chat:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Chat not found")

    if not user_can_edit_chat(chat, user):
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="Access denied")
    return chat_repo.update(session, chat, chat_update)


@router.delete("/chats/{chat_id}")
def delete_chat(session: SessionDep, user: CurrentUserDep, chat_id: UUID):
    chat = chat_repo.get(session, chat_id)
    if not chat:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Chat not found")

    if not user_can_edit_chat(chat, user):
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="Access denied")
    return chat_repo.delete(session, chat)


class SubgraphResponse(BaseModel):
    entities: List[dict]
    relationships: List[dict]


@router.get(
    "/chat-messages/{chat_message_id}/subgraph", response_model=SubgraphResponse
)
def get_chat_subgraph(session: SessionDep, user: OptionalUserDep, chat_message_id: int):
    chat_message = chat_repo.get_message(session, chat_message_id)
    if not chat_message:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Chat message not found"
        )

    if not user_can_view_chat(chat_message.chat, user):
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="Access denied")

    entities, relations = get_chat_message_subgraph(session, chat_message)

    return SubgraphResponse(entities=entities, relationships=relations)


@router.get(
    "/chat-messages/{chat_message_id}/recommended-questions", response_model=List[str]
)
def get_recommended_questions(session: SessionDep, chat_message_id: int):
    chat_message = chat_repo.get_message(session, chat_message_id)
    if not chat_message or len(chat_message.content) == 0:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Chat message not found"
        )

    return get_chat_message_recommend_questions(session, chat_message)


@router.post(
    "/chat-messages/{chat_message_id}/recommended-questions", response_model=List[str]
)
def refresh_recommended_questions(session: SessionDep, chat_message_id: int):
    chat_message = chat_repo.get_message(session, chat_message_id)
    if not chat_message or len(chat_message.content) == 0:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Chat message not found"
        )

    remove_chat_message_recommend_questions(session, chat_message_id)
    return get_chat_message_recommend_questions(session, chat_message)
