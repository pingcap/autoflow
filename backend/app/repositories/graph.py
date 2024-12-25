from typing import Type

from sqlmodel import Session, select, func, delete, SQLModel

from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.chunk import get_kb_chunk_model
from app.models.entity import get_kb_entity_model
from app.models.relationship import get_kb_relationship_model


class GraphRepo:
    def __init__(
        self,
        entity_model: Type[SQLModel],
        relationship_model: Type[SQLModel],
        chunk_model: Type[SQLModel],
    ):
        self.entity_model = entity_model
        self.relationship_model = relationship_model
        self.chunk_model = chunk_model

    def count_entities(self, session: Session):
        return session.scalar(select(func.count(self.entity_model.id)))

    def count_relationships(self, session: Session):
        return session.scalar(select(func.count(self.relationship_model.id)))

    def delete_orphaned_entities(self, session: Session):
        orphaned_entity_ids = (
            select(self.entity_model.id)
            .outerjoin(
                self.relationship_model,
                (self.relationship_model.target_entity_id == self.entity_model.id)
                | (self.relationship_model.source_entity_id == self.entity_model.id),
            )
            .where(self.relationship_model.id.is_(None))
            .scalar_subquery()
        )
        stmt = delete(self.entity_model).where(
            self.entity_model.id.in_(orphaned_entity_ids)
        )
        session.exec(stmt)

    def delete_data_source_relationships(self, session: Session, datasource_id: int):
        doc_ids_subquery = select(Document.id).where(
            Document.data_source_id == datasource_id
        )
        chunk_ids_subquery = select(self.chunk_model.id).where(
            self.chunk_model.document_id.in_(doc_ids_subquery)
        )
        stmt = delete(self.relationship_model).where(
            self.relationship_model.chunk_id.in_(chunk_ids_subquery)
        )
        session.exec(stmt)

    def delete_document_relationships(self, session: Session, document_id: int):
        chunk_ids_subquery = select(self.chunk_model.id).where(
            self.chunk_model.document_id == document_id
        )
        stmt = delete(self.relationship_model).where(
            self.relationship_model.chunk_id.in_(chunk_ids_subquery)
        )
        session.exec(stmt)


def get_kb_graph_repo(kb: KnowledgeBase) -> GraphRepo:
    chunk_model = get_kb_chunk_model(kb)
    entity_model = get_kb_entity_model(kb)
    relationship_model = get_kb_relationship_model(kb)
    return GraphRepo(entity_model, relationship_model, chunk_model)
