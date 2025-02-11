from fastapi import APIRouter, Depends
from fastapi_pagination import Params


from app.repositories.user import user_repo
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.api.admin_routes.models import (
    UserDescriptor,
)

router = APIRouter(
    prefix="/admin/users",
    tags=["admin/user"],
)


@router.get("/search")
def search_users(
    session: SessionDep,
    user: CurrentSuperuserDep,
    search: str | None = None,
    params: Params = Depends(),
) -> list[UserDescriptor]:
    return user_repo.search_users(session, search, params)
