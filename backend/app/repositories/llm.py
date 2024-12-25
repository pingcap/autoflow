from typing import Type, Optional

from fastapi import Depends
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy import update
from sqlmodel import select, Session

from app.exceptions import DefaultLLMNotFound, LLMNotFound
from app.models import LLM as DBLLM
from app.repositories.base_repo import BaseRepo


class LLMRepo(BaseRepo):
    model_cls: DBLLM

    def paginate(self, session: Session, params: Params = Depends()) -> Page[DBLLM]:
        query = select(DBLLM)
        # Make sure the default llm is always on top.
        query = query.order_by(DBLLM.is_default.desc(), DBLLM.created_at.desc())
        return paginate(session, query, params)

    def get(self, session: Session, llm_id: int) -> Optional[DBLLM]:
        return session.get(DBLLM, llm_id)

    def must_get(self, session: Session, llm_id: int) -> Type[DBLLM]:
        db_llm = self.get(session, llm_id)
        if db_llm is None:
            raise LLMNotFound(llm_id)
        return db_llm

    def create(self, session: Session, llm: DBLLM) -> DBLLM:
        # If there is no exiting model, the first model is
        # automatically set as the default model.
        if not self.exists_any_model(session):
            llm.is_default = True

        if llm.is_default:
            self.unset_default_model(session)

        llm.id = None
        session.add(llm)
        session.commit()
        session.refresh(llm)

        return llm

    def exists_any_model(self, session: Session) -> bool:
        stmt = select(DBLLM).with_for_update().limit(1)
        return session.exec(stmt).one_or_none() is not None

    # Default model

    def get_default(self, session: Session) -> Type[DBLLM] | None:
        stmt = (
            select(DBLLM)
            .where(DBLLM.is_default == True)
            .order_by(DBLLM.updated_at.desc())
            .limit(1)
        )
        return session.exec(stmt).first()

    def has_default(self, session: Session) -> bool:
        return self.get_default(session) is not None

    def must_get_default(self, session: Session) -> Type[DBLLM]:
        db_llm = self.get_default(session)
        if db_llm is None:
            raise DefaultLLMNotFound()
        return db_llm

    def unset_default_model(self, session: Session):
        session.exec(update(DBLLM).values(is_default=False))

    def set_default_model(self, session: Session, new_default_model_id: int):
        self.unset_default_model(session)
        session.exec(
            update(DBLLM)
            .where(DBLLM.id == new_default_model_id)
            .values(is_default=True)
        )
        session.commit()


llm_repo = LLMRepo()
