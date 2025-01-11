from typing import Type

from sqlalchemy import func, delete
from sqlmodel import Session, select, SQLModel
from app.repositories.base_repo import BaseRepo

from app.models import (
    Chunk as DBChunk,
    Document as DBDocument,
)


class ChunkRepo(BaseRepo):
    def __init__(self, chunk_model: Type[SQLModel] = DBChunk):
        self.model_cls = chunk_model

    def document_exists_chunks(self, session: Session, document_id: int) -> bool:
        return (
            session.exec(
                select(self.model_cls).where(self.model_cls.document_id == document_id)
            ).first()
            is not None
        )

    def get_documents_by_chunk_ids(
        self, session: Session, chunk_ids: list[str]
    ) -> list[DBDocument]:
        stmt = select(DBDocument).where(
            DBDocument.id.in_(
                select(self.model_cls.document_id).where(
                    self.model_cls.id.in_(chunk_ids),
                )
            ),
        )
        return list(session.exec(stmt).all())

    def get_document_chunks(self, session: Session, document_id: int):
        return session.exec(
            select(self.model_cls).where(self.model_cls.document_id == document_id)
        ).all()

    def count(self, session: Session):
        return session.scalar(select(func.count(self.model_cls.id)))

    def delete_by_datasource(self, session: Session, datasource_id: int):
        doc_ids_subquery = select(DBDocument.id).where(
            DBDocument.data_source_id == datasource_id
        )
        stmt = delete(self.model_cls).where(
            self.model_cls.document_id.in_(doc_ids_subquery)
        )
        session.exec(stmt)

    def delete_by_document(self, session: Session, document_id: int):
        stmt = delete(self.model_cls).where(self.model_cls.document_id == document_id)
        session.exec(stmt)


chunk_repo = ChunkRepo()
