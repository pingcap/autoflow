from typing import Type, Optional

from sqlalchemy import func
from sqlmodel import Session, select, SQLModel
from app.repositories.base_repo import BaseRepo

from app.models import (
    Chunk as DBChunk,
)

class ChunkRepo(BaseRepo):

    def __init__(self, chunk_model: Optional[SQLModel] = DBChunk):
        self.model_cls = chunk_model

    def document_exists_chunks(self, session: Session, document_id: int) -> bool:
        return session.exec(
            select(self.model_cls).where(self.model_cls.document_id == document_id)
        ).first() is not None
    
    def get_document_chunks(
        self,
        session: Session,
        document_id: int
    ):
        return session.exec(select(self.model_cls).where(self.model_cls.document_id == document_id)).all()


    def count(self, session: Session):
        return session.scalar(select(func.count(self.model_cls.id)))

chunk_repo = ChunkRepo()
