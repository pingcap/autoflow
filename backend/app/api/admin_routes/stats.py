from datetime import date
from pydantic import BaseModel
from fastapi import APIRouter
from app.api.deps import CurrentSuperuserDep, SessionDep
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
