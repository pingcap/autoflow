from typing import Optional
from fastapi import APIRouter, Depends
from fastapi_pagination import Page, Params

from app.repositories.user import user_repo
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.api.admin_routes.models import (
    UserDescriptor,
)

router = APIRouter(
    prefix="/admin/users",
    tags=["admin/users"],
)


@router.get("/search")
def search_users(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    search: Optional[str] = None,
    params: Params = Depends(),
) -> Page[UserDescriptor]:
    return user_repo.search_users(db_session, search, params)
