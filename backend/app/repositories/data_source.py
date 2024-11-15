from typing import Optional
from datetime import datetime, UTC

from sqlmodel import select, Session, func, update
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate

from app.models import (
    DataSource,
    Document,
    DocIndexTaskStatus,
    Chunk,
    KgIndexStatus,
    Relationship,
)
from app.repositories.base_repo import BaseRepo
from app.api.admin_routes.knowledge_base.models import VectorIndexError, KGIndexError


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

    def overview(self, session: Session, data_source: DataSource) -> dict:
        data_source_id = data_source.id
        documents_count = session.scalar(
            select(func.count(Document.id)).where(
                Document.data_source_id == data_source_id
            )
        )
        chunks_count = session.scalar(
            select(func.count(Chunk.id)).where(
                Chunk.document.has(Document.data_source_id == data_source_id)
            )
        )

        statement = (
            select(Document.index_status, func.count(Document.id))
            .where(Document.data_source_id == data_source_id)
            .group_by(Document.index_status)
            .order_by(Document.index_status)
        )
        results = session.exec(statement).all()
        vector_index_status = {s: c for s, c in results}

        overview_data = {
            "documents": {
                "total": documents_count,
            },
            "chunks": {
                "total": chunks_count,
            },
            "vector_index": vector_index_status,
        }

        if data_source.build_kg_index:
            relationships_count = session.scalar(
                select(func.count(Relationship.id)).where(
                    Relationship.document_id.in_(
                        select(Document.id).where(
                            Document.data_source_id == data_source_id
                        )
                    )
                )
            )
            overview_data["relationships"] = {
                "total": relationships_count,
            }

            statement = (
                select(Chunk.index_status, func.count(Chunk.id))
                .where(Chunk.document.has(Document.data_source_id == data_source_id))
                .group_by(Chunk.index_status)
                .order_by(Chunk.index_status)
            )
            results = session.exec(statement).all()
            kg_index_status = {s: c for s, c in results}
            overview_data["kg_index"] = kg_index_status

        return overview_data

    def vector_index_built_errors(
        self,
        session: Session,
        data_source: DataSource,
        params: Params | None = Params(),
    ) -> Page[VectorIndexError]:
        query = (
            select(
                Document.id,
                Document.name,
                Document.source_uri,
                Document.index_result,
            )
            .where(
                Document.data_source_id == data_source.id,
                Document.index_status == DocIndexTaskStatus.FAILED,
            )
            .order_by(Document.id.desc())
        )
        return paginate(
            session,
            query,
            params,
            transformer=lambda items: [
                VectorIndexError(
                    document_id=item[0],
                    document_name=item[1],
                    source_uri=item[2],
                    error=item[3],
                )
                for item in items
            ],
        )

    def kg_index_built_errors(
        self,
        session: Session,
        data_source: DataSource,
        params: Params | None = Params(),
    ) -> Page[KGIndexError]:
        query = (
            select(
                Chunk.id,
                Chunk.source_uri,
                Chunk.index_result,
            )
            .where(
                Chunk.document.has(Document.data_source_id == data_source.id),
                Chunk.index_status == KgIndexStatus.FAILED,
            )
            .order_by(Chunk.id.desc())
        )
        return paginate(
            session,
            query,
            params,
            transformer=lambda items: [
                KGIndexError(
                    chunk_id=item[0],
                    source_uri=item[1],
                    error=item[2],
                )
                for item in items
            ],
        )

    def set_failed_vector_index_tasks_to_pending(
        self, session: Session, data_source: DataSource
    ) -> list[int]:
        failed_document_ids = session.exec(
            select(Document.id).where(
                Document.data_source_id == data_source.id,
                Document.index_status == DocIndexTaskStatus.FAILED,
            )
        ).all()
        stmt = (
            update(Document)
            .where(
                Document.id.in_(failed_document_ids),
                Document.index_status == DocIndexTaskStatus.FAILED,
            )
            .values(index_status=DocIndexTaskStatus.PENDING)
        )
        session.exec(stmt)
        session.commit()
        return failed_document_ids

    def set_failed_kg_index_tasks_to_pending(
        self, session: Session, data_source: DataSource
    ) -> list[tuple]:
        failed_chunks = session.exec(
            select(
                Chunk.id,
                Chunk.document_id,
            ).where(
                Chunk.document.has(Document.data_source_id == data_source.id),
                Chunk.index_status == KgIndexStatus.FAILED,
            )
        ).all()
        failed_chunks_ids = [chunk[0] for chunk in failed_chunks]
        stmt = (
            update(Chunk)
            .where(Chunk.id.in_(failed_chunks_ids))
            .values(index_status=KgIndexStatus.PENDING)
        )
        session.exec(stmt)
        session.commit()
        return failed_chunks


data_source_repo = DataSourceRepo()
