from fastapi_pagination import Page, Params, paginate
from sqlmodel import Session
from app.models.auth import User
from app.repositories.base_repo import BaseRepo
from sqlalchemy import select


class UserRepo(BaseRepo):
    model_cls: User

    def search_users(
        self,
        session: Session,
        search: str | None = None,
        params: Params | None = Params(),
    ) -> Page[User]:
        query = select(User)

        if search:
            query = query.where(User.email.ilike(f"%{search}%"))

        query = query.order_by(User.id)
        return paginate(session, query, params)


user_repo = UserRepo()
