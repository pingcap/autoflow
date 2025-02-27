from datetime import datetime
from typing import Dict, Optional

from sqlalchemy import JSON, func
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlmodel import (
    Field,
    Column,
    DateTime,
    Relationship as SQLRelationship,
    SQLModel,
)

from autoflow.db_models import DataSource
from autoflow.knowledge_base.config import GeneralChunkingConfig, IndexMethod

# For compatibility with old code, define a fake knowledge base id.
PHONY_KNOWLEDGE_BASE_ID = 0


class KnowledgeBaseDataSource(SQLModel, table=True):
    __tablename__ = "knowledge_base_datasources"

    knowledge_base_id: int = Field(primary_key=True, foreign_key="knowledge_bases.id")
    data_source_id: int = Field(primary_key=True, foreign_key="data_sources.id")


class KnowledgeBase(SQLModel, table=True):
    __tablename__ = "knowledge_bases"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=255, nullable=False)
    description: Optional[str] = Field(sa_column=Column(MEDIUMTEXT), default=None)
    index_methods: list[IndexMethod] = Field(
        default=[IndexMethod.VECTOR_SEARCH], sa_column=Column(JSON)
    )

    # The config for chunking, the process to break down the document into smaller chunks.
    chunking_config: Dict = Field(
        sa_column=Column(JSON), default=GeneralChunkingConfig().model_dump()
    )

    # Data sources config.
    data_sources: list[DataSource] = SQLRelationship(link_model=KnowledgeBaseDataSource)

    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
        ),
    )
    deleted_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )

    def __hash__(self):
        return hash(self.id)
