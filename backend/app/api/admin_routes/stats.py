from datetime import date
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate

from sqlmodel import select

from app.api.deps import CurrentSuperuserDep, SessionDep
from app.repositories import chat_repo
from app.models import Feedback
from app.models.base import UUIDBaseModel

from app.repositories import chat_repo

router = APIRouter()


class DateRangeStats(BaseModel):
    start_date: date
    end_date: date


class ChatStats(DateRangeStats):
    values: list


@router.get("/admin/stats/trend/chat-user")
def chat_count_trend(
    session: SessionDep, user: CurrentSuperuserDep, start_date: date, end_date: date
) -> ChatStats:
    stats = chat_repo.chat_trend_by_user(session, start_date, end_date)
    return ChatStats(start_date=start_date, end_date=end_date, values=stats)


@router.get("/admin/stats/trend/chat-origin")
def chat_origin_trend(
    session: SessionDep, user: CurrentSuperuserDep, start_date: date, end_date: date
) -> ChatStats:
    stats = chat_repo.chat_trend_by_origin(session, start_date, end_date)
    return ChatStats(start_date=start_date, end_date=end_date, values=stats)


class ChatOrigin(UUIDBaseModel):
    origin: str


@router.get("/admin/stats/chats/origins")
def list_chat_origins(
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> list[ChatOrigin]:
    chat_origins = []
    # chats = session.exec(select(Chat.origin, Chat.id).order_by(Chat.created_at.desc()))
    for chat in chat_repo.list_chat_origins(session):
        chat_origins.append(
            ChatOrigin(
                id=chat.id,
                origin=chat.origin,
            )
        )
    return chat_origins


@router.get("/admin/stats/feedbacks/origins")
def list_feedback_origins(
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> Page[str]:
    return paginate(
        session,
        select(Feedback.origin).distinct().order_by(Feedback.origin.asc()),
        params,
    )
