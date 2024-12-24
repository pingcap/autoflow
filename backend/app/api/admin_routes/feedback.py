from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi_pagination import Params, Page

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.models import AdminFeedbackPublic, FeedbackFilters
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
