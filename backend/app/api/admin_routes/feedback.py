from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlmodel import select

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.models import AdminFeedbackPublic, FeedbackFilters, Feedback
from app.repositories import feedback_repo

router = APIRouter()


@router.get("/admin/feedbacks")
def list_feedbacks(
    session: SessionDep,
    user: CurrentSuperuserDep,
    filters: Annotated[FeedbackFilters, Query()],
    params: Params = Depends(),
) -> Page[AdminFeedbackPublic]:
    return feedback_repo.paginate(
        session=session,
        filters=filters,
        params=params,
    )
    # return paginate(
    #     session,
    #     select(Feedback).order_by(Feedback.created_at.desc()),
    #     params,
    #     transformer=lambda items: [
    #         AdminFeedbackPublic(
    #             **item.model_dump(),
    #             chat_title=item.chat.title,
    #             chat_origin=item.chat.origin,
    #             chat_message_content=item.chat_message.content,
    #             user_email=item.user.email if item.user else None,
    #         )
    #         for item in items
    #     ],
    # )

@router.get("/admin/feedback_origins")
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

