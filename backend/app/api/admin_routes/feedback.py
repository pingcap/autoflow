from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query
from fastapi_pagination import Params, Page

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.models import AdminFeedbackPublic, FeedbackFilters
from app.models.feedback import FeedbackOrigin
from app.repositories import feedback_repo

router = APIRouter(
    prefix="/admin/feedbacks",
    tags=["admin/feedback"],
)


@router.get("/")
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


@router.get("/origins")
def list_feedback_origins(
    session: SessionDep,
    user: CurrentSuperuserDep,
    search: Optional[str] = None,
    params: Params = Depends(),
) -> Page[FeedbackOrigin]:
    return feedback_repo.list_feedback_origins(session, search, params)
