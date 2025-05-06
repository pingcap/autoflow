from typing import Type, Optional

from fastapi import Depends
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy import update
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import select, Session

from app.exceptions import DefaultLLMNotFound, LLMNotFound
from app.models import LLM, LLMUpdate
from app.repositories.base_repo import BaseRepo


class LLMRepo(BaseRepo):
    model_cls: LLM

    def paginate(self, session: Session, params: Params = Depends()) -> Page[LLM]:
        query = select(LLM)
        # Make sure the default llm is always on top.
        query = query.order_by(LLM.is_default.desc(), LLM.created_at.desc())
        return paginate(session, query, params)

    def get(self, session: Session, llm_id: int) -> Optional[LLM]:
        return session.get(LLM, llm_id)

    def must_get(self, session: Session, llm_id: int) -> LLM:
        db_llm = self.get(session, llm_id)
        if db_llm is None:
            raise LLMNotFound(llm_id)
        return db_llm

    def create(self, session: Session, llm: LLM) -> LLM:
        # If there is no exiting model, the first model is
        # automatically set as the default model.
        if not self.exists_any_model(session):
            llm.is_default = True

        llm.id = None
        session.add(llm)
        session.commit()
        session.refresh(llm)

        return llm

    def exists_any_model(self, session: Session) -> bool:
        stmt = select(LLM).with_for_update().limit(1)
        return session.exec(stmt).one_or_none() is not None

    # Default model

    def get_default(self, session: Session) -> Type[LLM] | None:
        stmt = (
            select(LLM)
            .where(LLM.is_default == True)
            .order_by(LLM.updated_at.desc())
            .limit(1)
        )
        return session.exec(stmt).first()

    def has_default(self, session: Session) -> bool:
        return self.get_default(session) is not None

    def must_get_default(self, session: Session) -> Type[LLM]:
        db_llm = self.get_default(session)
        if db_llm is None:
            raise DefaultLLMNotFound()
        return db_llm

    def _unset_default(self, session: Session):
        session.exec(update(LLM).values(is_default=False))

    def set_default(self, session: Session, llm: LLM) -> LLM:
        self._unset_default(session)
        llm.is_default = True
        session.add(llm)
        session.commit()
        session.refresh(llm)
        return llm

    def update(self, session: Session, llm: LLM, llm_update: LLMUpdate) -> LLM:
        for field, value in llm_update.model_dump(exclude_unset=True).items():
            setattr(llm, field, value)
            flag_modified(llm, field)

        session.add(llm)
        session.commit()
        session.refresh(llm)
        return llm


llm_repo = LLMRepo()
