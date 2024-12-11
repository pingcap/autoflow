import enum
from datetime import datetime
from typing import Any, Optional, Type
from uuid import UUID

from sqlalchemy import DateTime, func
from sqlmodel import (
    Field,
    Column,
    Text,
    JSON,
    Relationship as SQLRelationship,
    SQLModel,
)
from tidb_vector.sqlalchemy import VectorType
from llama_index.core.schema import TextNode

from app.core.config import settings
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from .base import UpdatableBaseModel, UUIDBaseModel
from app.models.knowledge_base_scoped.registry import get_kb_scoped_registry
from .knowledge_base_scoped.table_naming import (
    get_kb_chunks_table_name,
    get_kb_vector_dims,
)
from app.models.patch.sql_model import SQLModel as PatchSQLModel
from ..utils.uuid6 import uuid7


class KgIndexStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Notice: DO NOT forget to modify the definition in `get_kb_chunk_model` to
# keep the table structure on both sides consistent.
class Chunk(UUIDBaseModel, UpdatableBaseModel, table=True):
    hash: str = Field(max_length=64)
    text: str = Field(sa_column=Column(Text))
    meta: dict | list = Field(default={}, sa_column=Column(JSON))
    embedding: Any = Field(
        sa_column=Column(
            VectorType(settings.EMBEDDING_DIMS), comment="hnsw(distance=cosine)"
        )
    )
    document_id: int = Field(foreign_key="documents.id", nullable=True)
    document: "Document" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Chunk.document_id == Document.id",
        },
    )
    relations: dict | list = Field(default={}, sa_column=Column(JSON))
    source_uri: str = Field(max_length=512, nullable=True)

    # TODO: Add vector_index_status, vector_index_result column, vector index should be optional in the future.

    # TODO: Rename to kg_index_status, kg_index_result column.
    index_status: KgIndexStatus = KgIndexStatus.NOT_STARTED
    index_result: str = Field(sa_column=Column(Text, nullable=True))

    __tablename__ = "chunks"

    def to_llama_text_node(self) -> TextNode:
        return TextNode(
            id_=self.id.hex,
            text=self.text,
            embedding=list(self.embedding),
            metadata=self.meta,
        )


def get_kb_chunk_model(kb: KnowledgeBase) -> Type[SQLModel]:
    vector_dimension = get_kb_vector_dims(kb)
    chunks_table_name = get_kb_chunks_table_name(kb)
    ctx = get_kb_scoped_registry(kb)

    if ctx.chunk_model:
        return ctx.chunk_model

    class KBChunk(PatchSQLModel, table=True, registry=ctx.registry):
        __tablename__ = chunks_table_name
        __table_args__ = {"extend_existing": True}

        id: UUID = Field(
            primary_key=True, index=True, nullable=False, default_factory=uuid7
        )
        hash: str = Field(max_length=64)
        text: str = Field(sa_column=Column(Text))
        meta: dict | list = Field(default={}, sa_column=Column(JSON))
        embedding: Any = Field(
            sa_column=Column(
                VectorType(vector_dimension), comment="hnsw(distance=cosine)"
            )
        )
        document_id: int = Field(foreign_key="documents.id", nullable=True)
        document: "Document" = SQLRelationship()
        relations: dict | list = Field(default={}, sa_column=Column(JSON))
        source_uri: str = Field(max_length=512, nullable=True)

        # TODO: Add vector_index_status, vector_index_result column, vector index should be optional in the future.

        # TODO: Rename to kg_index_status, kg_index_result column.
        index_status: KgIndexStatus = KgIndexStatus.NOT_STARTED
        index_result: str = Field(sa_column=Column(Text, nullable=True))

        created_at: Optional[datetime] = Field(
            default=None,
            sa_column=Column(DateTime(timezone=True), server_default=func.now()),
        )
        updated_at: Optional[datetime] = Field(
            default=None,
            sa_type=DateTime(timezone=True),
            sa_column_kwargs={"server_default": func.now(), "onupdate": func.now()},
        )

        def to_llama_text_node(self) -> TextNode:
            return TextNode(
                id_=self.id.hex,
                text=self.text,
                embedding=list(self.embedding),
                metadata=self.meta,
            )

    ctx.chunk_model = KBChunk
    return KBChunk
