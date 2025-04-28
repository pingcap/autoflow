from typing import Optional
from datetime import datetime, UTC

from sqlmodel import select, Session
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate

from app.models import (
    DataSource,
)
from app.repositories.base_repo import BaseRepo


class DataSourceRepo(BaseRepo):
    model_cls = DataSource

    def paginate(
        self,
        session: Session,
        params: Params | None = Params(),
    ) -> Page[DataSource]:
        query = (
            select(DataSource)
            .where(DataSource.deleted_at == None)
            .order_by(DataSource.created_at.desc())
        )
        return paginate(session, query, params)

    def get(
        self,
        session: Session,
        data_source_id: int,
    ) -> Optional[DataSource]:
        return session.exec(
            select(DataSource).where(
                DataSource.id == data_source_id, DataSource.deleted_at == None
            )
        ).first()

    def delete(self, session: Session, data_source: DataSource) -> None:
        data_source.deleted_at = datetime.now(UTC)
        session.add(data_source)
        session.commit()


data_source_repo = DataSourceRepo()
