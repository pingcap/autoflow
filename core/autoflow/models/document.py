import enum
from typing import Optional, Dict
from datetime import datetime
from uuid import UUID

from llama_index.core.schema import Document as LlamaDocument
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlmodel import (
    Field,
    Column,
    DateTime,
    JSON,
    SQLModel,
)


class DocumentIndexStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    CHUNKING = "chunking"
    COMPLETED = "completed"
    FAILED = "failed"


class VectorSearchIndexStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class KnowledgeGraphIndexStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: Optional[int] = Field(default=None, primary_key=True)
    hash: str = Field(max_length=32)
    name: str = Field(max_length=256)
    content: str = Field(sa_column=Column(MEDIUMTEXT))
    mime_type: str = Field(max_length=100)
    meta: Optional[Dict] = Field(default={}, sa_column=Column(JSON))
    data_source_id: UUID = Field(nullable=True)
    knowledge_base_id: UUID = Field(nullable=True)
    created_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    updated_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))

    def __hash__(self) -> int:
        return hash(self.hash)

    def to_llama_document(self) -> LlamaDocument:
        return LlamaDocument(
            id_=str(self.id),
            text=self.content,
            metadata=self.meta,
        )
