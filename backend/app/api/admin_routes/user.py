from fastapi import APIRouter, Depends
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlmodel import select

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.models import User

router = APIRouter()


@router.get("/admin/users")
def list_users(
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> Page[User]:
    return paginate(
        session,
        select(User).order_by(User.id),
        params=params,
    )
