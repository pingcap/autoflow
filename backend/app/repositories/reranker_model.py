from typing import Type, Optional

from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy import update
from sqlmodel import select, Session

from app.exceptions import RerankerModelNotFound, DefaultRerankerModelNotFound
from app.models import RerankerModel
from app.repositories.base_repo import BaseRepo


class RerankerModelRepo(BaseRepo):
    model_cls: RerankerModel

    def paginate(
        self,
        session: Session,
        params: Params | None  = Params()
    ) -> Page[RerankerModel]:
        query = select(RerankerModel)
        # Make sure the default reranker model is always on top.
        query = query.order_by(RerankerModel.is_default.desc(), RerankerModel.created_at.desc())
        return paginate(session, query, params)

    def get(self, session: Session, reranker_model_id: int) -> Optional[RerankerModel]:
        return session.get(RerankerModel, reranker_model_id)

    def must_get(self, session: Session, reranker_model_id: int) -> Type[RerankerModel]:
        db_reranker_model = self.get(session, reranker_model_id)
        if db_reranker_model is None:
            raise RerankerModelNotFound(reranker_model_id)
        return db_reranker_model

    def get_default(self, session: Session) -> Optional[RerankerModel]:
        stmt = (
            select(RerankerModel)
                .where(RerankerModel.is_default == True)
                .order_by(RerankerModel.updated_at.desc())
                .limit(1)
        )
        return session.exec(stmt).first()

    def must_get_default(self, session: Session) -> RerankerModel:
        db_reranker_model = self.get_default(session)
        if db_reranker_model is None:
            raise DefaultRerankerModelNotFound()
        return db_reranker_model

    def create(self, session: Session, reranker_model: RerankerModel) -> RerankerModel:
        # If there is no exiting model, the first model is
        # automatically set as the default model.
        if not self.exists_any_model(session):
            reranker_model.is_default = True

        if reranker_model.is_default:
            session.exec(update(RerankerModel).values(is_default=False))

        session.add(reranker_model)
        session.commit()
        session.refresh(reranker_model)
        return reranker_model

        return llm

    def exists_any_model(self, session: Session) -> bool:
        stmt = select(RerankerModel).with_for_update().limit(1)
        return session.exec(stmt).one_or_none() is not None


reranker_model_repo = RerankerModelRepo()