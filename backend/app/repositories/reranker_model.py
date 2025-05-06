from typing import Optional

from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy import update
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select

from app.exceptions import RerankerModelNotFound, DefaultRerankerModelNotFound
from app.models import RerankerModel
from app.models.chat_engine import ChatEngine
from app.models.reranker_model import RerankerModelUpdate
from app.repositories.base_repo import BaseRepo


class RerankerModelRepo(BaseRepo):
    model_cls: RerankerModel

    def paginate(
        self, session: Session, params: Params | None = Params()
    ) -> Page[RerankerModel]:
        query = select(RerankerModel)
        # Make sure the default reranker model is always on top.
        query = query.order_by(
            RerankerModel.is_default.desc(), RerankerModel.created_at.desc()
        )
        return paginate(session, query, params)

    def get(self, session: Session, model_id: int) -> Optional[RerankerModel]:
        return session.get(RerankerModel, model_id)

    def must_get(self, session: Session, model_id: int) -> RerankerModel:
        db_model = self.get(session, model_id)
        if db_model is None:
            raise RerankerModelNotFound(model_id)
        return db_model

    def exists_any_model(self, session: Session) -> bool:
        stmt = select(RerankerModel).with_for_update().limit(1)
        return session.exec(stmt).one_or_none() is not None

    def create(self, session: Session, reranker_model: RerankerModel) -> RerankerModel:
        # If there is no exiting model, the first model will be
        # set as the default model.
        if not self.exists_any_model(session):
            reranker_model.is_default = True

        if reranker_model.is_default:
            self.unset_default(session)

        reranker_model.id = None
        session.add(reranker_model)
        session.commit()
        session.refresh(reranker_model)

        return reranker_model

    def update(
        self,
        session: Session,
        reranker_model: RerankerModel,
        model_update: RerankerModelUpdate,
    ) -> RerankerModel:
        for field, value in model_update.model_dump(exclude_unset=True).items():
            setattr(reranker_model, field, value)
            flag_modified(reranker_model, field)

        session.commit()
        session.refresh(reranker_model)
        return reranker_model

    def delete(self, db_session: Session, reranker_model: RerankerModel):
        # TODO: Support to specify a new reranker model to replace the current reranker model.
        db_session.exec(
            update(ChatEngine)
            .where(ChatEngine.reranker_id == reranker_model.id)
            .values(reranker_id=None)
        )

        db_session.delete(reranker_model)
        db_session.commit()

    # Default model

    def get_default(self, session: Session) -> Optional[RerankerModel]:
        stmt = select(RerankerModel).where(RerankerModel.is_default == True).limit(1)
        return session.exec(stmt).first()

    def has_default(self, session: Session) -> bool:
        return self.get_default(session) is not None

    def must_get_default(self, session: Session) -> RerankerModel:
        db_reranker_model = self.get_default(session)
        if db_reranker_model is None:
            raise DefaultRerankerModelNotFound()
        return db_reranker_model

    def unset_default(self, session: Session):
        session.exec(update(RerankerModel).values(is_default=False))

    def set_default(self, session: Session, model: RerankerModel):
        self.unset_default(session)
        model.is_default = True
        flag_modified(model, "is_default")
        session.commit()
        session.refresh(model)
        return model


reranker_model_repo = RerankerModelRepo()
