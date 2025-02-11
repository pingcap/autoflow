from typing import Optional
from fastapi_pagination import Page, Params
from fastapi_pagination.ext.sqlmodel import paginate
from sqlmodel import Session, select
from app.models.auth import User
from app.repositories.base_repo import BaseRepo


class UserRepo(BaseRepo):
    model_cls: User

    def search_users(
        self,
        db_session: Session,
        search: Optional[str] = None,
        params: Params = Params(),
    ) -> Page[User]:
        query = select(User)

        if search:
            query = query.where(User.email.ilike(f"%{search}%"))

        query = query.order_by(User.id)
        return paginate(
            db_session,
            query,
            params,
        )


user_repo = UserRepo()
