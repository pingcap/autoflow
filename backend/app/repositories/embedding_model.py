from typing import Optional, Type

from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select, update

from app.api.admin_routes.embedding_model.models import (
    EmbeddingModelUpdate,
    EmbeddingModelCreate,
)
from app.exceptions import DefaultEmbeddingModelNotFound, EmbeddingModelNotFound
from app.models import EmbeddingModel
from app.models.knowledge_base import KnowledgeBase
from app.repositories.base_repo import BaseRepo


class EmbeddingModelRepo(BaseRepo):
    model_cls = EmbeddingModel

    def paginate(
        self, session: Session, params: Params | None = Params()
    ) -> Page[EmbeddingModel]:
        query = select(EmbeddingModel)
        # Make sure the default model is always on top.
        query = query.order_by(
            EmbeddingModel.is_default.desc(), EmbeddingModel.created_at.desc()
        )
        return paginate(session, query, params)

    def get(self, session: Session, model_id: int) -> Optional[EmbeddingModel]:
        return session.get(EmbeddingModel, model_id)

    def must_get(self, session: Session, model_id: int) -> Type[EmbeddingModel]:
        db_embed_model = self.get(session, model_id)
        if db_embed_model is None:
            raise EmbeddingModelNotFound(model_id)
        return db_embed_model

    def exists_any_model(self, session: Session) -> bool:
        stmt = select(EmbeddingModel).with_for_update().limit(1)
        return session.exec(stmt).one_or_none() is not None

    def create(self, session: Session, create: EmbeddingModelCreate):
        # If there is currently no model, the first model will be
        # set as the default model.
        if not self.exists_any_model(session):
            create.is_default = True

        if create.is_default:
            self._unset_default(session)

        embed_model = EmbeddingModel(
            name=create.name,
            provider=create.provider,
            model=create.model,
            vector_dimension=create.vector_dimension,
            config=create.config,
            credentials=create.credentials,
            is_default=create.is_default,
        )
        session.add(embed_model)
        session.commit()
        session.refresh(embed_model)

        return embed_model

    def update(
        self,
        session: Session,
        embed_model: EmbeddingModel,
        partial_update: EmbeddingModelUpdate,
    ) -> EmbeddingModel:
        for field, value in partial_update.model_dump(exclude_unset=True).items():
            setattr(embed_model, field, value)
            flag_modified(embed_model, field)

        session.commit()
        session.refresh(embed_model)
        return embed_model

    def delete(self, session: Session, model: EmbeddingModel):
        # TODO: Support to specify a new embedding model to replace the current embedding model.
        session.exec(
            update(KnowledgeBase)
            .where(KnowledgeBase.embedding_model_id == model.id)
            .values(embedding_model_id=None)
        )

        session.delete(model)
        session.commit()

    # Default model

    def get_default(self, session: Session) -> Type[EmbeddingModel]:
        stmt = select(EmbeddingModel).where(EmbeddingModel.is_default == True).limit(1)
        return session.exec(stmt).first()

    def has_default(self, session: Session) -> bool:
        return self.get_default(session) is not None

    def must_get_default(self, session: Session) -> Type[EmbeddingModel]:
        embed_model = self.get_default(session)
        if embed_model is None:
            raise DefaultEmbeddingModelNotFound()
        return embed_model

    def _unset_default(self, session: Session):
        session.exec(
            update(EmbeddingModel)
            .values(is_default=False)
            .where(EmbeddingModel.is_default == True)
        )

    def set_default(self, session: Session, model: EmbeddingModel):
        self._unset_default(session)
        model.is_default = True
        flag_modified(model, "is_default")
        session.commit()
        session.refresh(model)
        return model


embedding_model_repo = EmbeddingModelRepo()
