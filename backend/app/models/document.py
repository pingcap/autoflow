import enum
from typing import Optional
from datetime import datetime

from llama_index.core.schema import Document as LlamaDocument
from pydantic import ConfigDict
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlmodel import (
    Field,
    Column,
    Text,
    DateTime,
    JSON,
    String,
    Relationship as SQLRelationship,
)

from .base import UpdatableBaseModel
from app.types import MimeTypes


class DocIndexTaskStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(UpdatableBaseModel, table=True):
    # Avoid "expected `enum` but got `str`" error.
    model_config = ConfigDict(use_enum_values=True)

    id: Optional[int] = Field(default=None, primary_key=True)
    hash: str = Field(max_length=32)
    name: str = Field(max_length=256)
    content: str = Field(sa_column=Column(MEDIUMTEXT))
    mime_type: MimeTypes = Field(sa_column=Column(String(128), nullable=False))
    source_uri: str = Field(max_length=512)
    meta: dict | list = Field(default={}, sa_column=Column(JSON))
    # the last time the document was modified in the source system
    last_modified_at: Optional[datetime] = Field(sa_column=Column(DateTime))

    # TODO: rename to vector_index_status, vector_index_result.
    index_status: DocIndexTaskStatus = DocIndexTaskStatus.NOT_STARTED
    index_result: str = Field(sa_column=Column(Text, nullable=True))

    # TODO: add kg_index_status, kg_index_result column, unify the index status.

    data_source_id: int = Field(foreign_key="data_sources.id", nullable=True)
    data_source: "DataSource" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Document.data_source_id == DataSource.id",
        },
    )

    knowledge_base_id: int = Field(foreign_key="knowledge_bases.id", nullable=True)
    knowledge_base: "KnowledgeBase" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Document.knowledge_base_id == KnowledgeBase.id",
        },
    )

    __tablename__ = "documents"

    def to_llama_document(self) -> LlamaDocument:
        return LlamaDocument(
            id_=str(self.id),
            text=self.content,
            metadata=self.meta,
        )
