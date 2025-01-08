from fastapi import APIRouter, Depends
from fastapi_pagination import Params
from sqlmodel import select

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.models import User

from app.api.admin_routes.models import (
    UserDescriptor,
)
from fuzzywuzzy import process

router = APIRouter()


@router.get("/admin/users/search")
def search_users(
    session: SessionDep,
    user: CurrentSuperuserDep,
    search: str | None = None,
    params: Params = Depends(),
) -> list[UserDescriptor]:
    users = []
    for user in session.exec(select(User).order_by(User.id)):
        users.append(UserDescriptor(id=user.id, email=user.email))
    # when search is empty, return all users
    if not search:
        return users
    # when search is not empty, filter users by email
    emails = [user.email for user in users]
    matches = process.extract(search, emails, limit=len(emails))

    threshold = 70
    matched_emails = {match[0] for match in matches if match[1] >= threshold}

    return [user for user in users if user.email in matched_emails]
