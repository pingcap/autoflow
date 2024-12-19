from typing import Type, Optional

from sqlalchemy import func, delete
from sqlmodel import Session, select, SQLModel
from app.repositories.base_repo import BaseRepo

from app.models import (
    Chunk as DBChunk,
    Document,
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

    def get_document_chunks(self, session: Session, document_id: int):
        return session.exec(
            select(self.model_cls).where(self.model_cls.document_id == document_id)
        ).all()

    def count(self, session: Session):
        return session.scalar(select(func.count(self.model_cls.id)))

    def delete_by_datasource(self, session: Session, datasource_id: int):
        doc_ids_subquery = select(Document.id).where(
            Document.data_source_id == datasource_id
        )
        stmt = delete(self.model_cls).where(
            self.model_cls.document_id.in_(doc_ids_subquery)
        )
        session.exec(stmt)

    def delete_by_document(self, session: Session, document_id: int):
        stmt = delete(self.model_cls).where(self.model_cls.document_id == document_id)
        session.exec(stmt)


chunk_repo = ChunkRepo()
