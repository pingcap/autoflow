import enum
from typing import Optional, Dict
from datetime import datetime
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlmodel import (
    Field,
    Column,
    DateTime,
    JSON,
    Relationship as SQLRelationship,
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
    meta: Optional[Dict] = Field(default={}, sa_column=Column(JSON))

    # Data source.
    data_source_id: int = Field(foreign_key="data_sources.id", nullable=True)
    data_source: "DataSource" = SQLRelationship(  # noqa:F821
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Document.data_source_id == DataSource.id",
        },
    )

    # Knowledge Base.
    knowledge_base_id: int = Field(foreign_key="knowledge_bases.id", nullable=True)
    knowledge_base: "KnowledgeBase" = SQLRelationship(  # noqa:F821
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Document.knowledge_base_id == KnowledgeBase.id",
        },
    )

    created_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    updated_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
