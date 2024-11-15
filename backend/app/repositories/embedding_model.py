from typing import Type

from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select, update, SQLModel

from app.api.admin_routes.embedding_model.models import EmbeddingModelUpdate, EmbeddingModelCreate
from app.exceptions import EmbeddingModelNotFoundError
from app.models import  EmbeddingModel
from app.repositories.base_repo import BaseRepo


class EmbeddingModelRepo(BaseRepo):
    model_cls = EmbeddingModel

    def create(self, session: Session, create: EmbeddingModelCreate):
        # If there is currently no model, the first model is
        # automatically set as the default model.
        if not embedding_model_repo.exists_any_model(session):
            create.is_default = True

        if create.is_default:
            embedding_model_repo.unset_default_model(session)

        embed_model = EmbeddingModel(
            name=create.name,
            provider=create.provider,
            model=create.model,
            vector_dimension=create.vector_dimension,
            config=create.config,
            credentials=create.credentials
        )
        session.add(embed_model)
        session.commit()
        session.refresh(embed_model)

        return embed_model

    def exists_any_model(self, session: Session) -> bool:
        stmt = select(EmbeddingModel).with_for_update().limit(1)
        return session.exec(stmt).one_or_none() is not None


    def must_get(self, session, model_id: int) -> Type[EmbeddingModel]:
        db_embed_model = self.get(session, model_id)
        if db_embed_model is None:
            raise EmbeddingModelNotFoundError(model_id)
        return db_embed_model


    def paginate(self, session: Session, params: Params | None = Params()) -> Page[EmbeddingModel]:
        query = (
            select(EmbeddingModel)
            .order_by(EmbeddingModel.created_at.desc())
        )
        return paginate(session, query, params)


    def get_default_model(self, session: Session) -> Type[EmbeddingModel]:
        stmt = (select(EmbeddingModel)
                .where(EmbeddingModel.is_default == True)
                .order_by(EmbeddingModel.updated_at.desc())
                .limit(1))
        return session.exec(stmt).first()


    def unset_default_model(self, session: Session):
        session.exec(
            update(EmbeddingModel)
            .values(is_default=False)
            .where(EmbeddingModel.is_default == True)
        )

    def set_default_model(self, session: Session, new_default_model_id: int):
        self.unset_default_model(session)
        session.exec(
            update(EmbeddingModel)
                .values(is_default=True)
                .where(EmbeddingModel.id == new_default_model_id)
        )
        session.commit()


    def update(
        self,
        session: Session,
        embed_model: EmbeddingModel,
        partial_update: EmbeddingModelUpdate,
    ) -> EmbeddingModel:
        set_default = partial_update.is_default
        if set_default:
            self.unset_default_model(session)

        for field, value in partial_update.model_dump(exclude_unset=True).items():
            setattr(embed_model, field, value)
            flag_modified(embed_model, field)

        session.commit()
        session.refresh(embed_model)
        return embed_model


embedding_model_repo = EmbeddingModelRepo()
