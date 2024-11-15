from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, field_validator

from app.api.admin_routes.models import EmbeddingModelDescriptor, LLMDescriptor, UserDescriptor
from app.exceptions import KBNoVectorIndexConfiguredError
from app.models import KgIndexStatus
from app.models.data_source import DataSourceType
from app.models.knowledge_base import IndexMethod


class CreateKBDataSourceRequest(BaseModel):
    id: Optional[int] = None
    name: str
    data_source_type: DataSourceType
    config: dict | list

    @field_validator("name")
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Please provide a name for the data source")
        return v

    # TODO: verify DataSource config.


class BaseKnowledgeBaseSetting(BaseModel):
    name: str
    description: str
    index_methods: list[IndexMethod]
    llm_id: Optional[int] = None
    data_sources: list[CreateKBDataSourceRequest]

    @field_validator("name")
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Please provide a name for the knowledge base")
        return v

    @field_validator("index_methods")
    def index_methods_must_has_vector(cls, v: list[IndexMethod]) -> list[IndexMethod]:
        # Notice: For now, knowledge base must be configured vector index method,
        # we will remove this limit in the feature.
        if IndexMethod.VECTOR not in v:
            raise KBNoVectorIndexConfiguredError()
        return v


class CreateKnowledgeBaseRequest(BaseKnowledgeBaseSetting):
    embedding_model_id: Optional[int] = None


class UpdateKnowledgeBaseRequest(BaseKnowledgeBaseSetting):
    pass


class KBDataSource(BaseModel):
    """
    Represents a linked data source for a knowledge base.
    """
    id: int
    name: str
    data_source_type: DataSourceType
    config: dict | list


class KnowledgeBaseDetail(BaseModel):
    """
    Represents a detailed view of a knowledge base.
    """
    id: int
    name: str
    description: str
    # Notice: By default, SQLModel will not serialize list type relationships.
    # https://github.com/fastapi/sqlmodel/issues/37#issuecomment-2093607242
    data_sources: list[KBDataSource]
    index_methods: list[IndexMethod]
    llm: LLMDescriptor | None = None
    embedding_model: EmbeddingModelDescriptor | None = None
    creator: UserDescriptor | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class KnowledgeBaseItem(BaseModel):
    """
    Represents a simplified view of a knowledge base for list display purposes.
    """
    id: int
    name: str
    description: str
    documents_total: int
    data_sources_total: int
    index_methods: list[IndexMethod]
    creator: UserDescriptor | None = None
    created_at: datetime
    updated_at: datetime


class VectorIndexError(BaseModel):
    document_id: int
    document_name: str
    source_uri: str
    error: str | None = None


class KGIndexError(BaseModel):
    document_id: int
    document_name: str
    source_uri: str
    chunk_id: UUID
    error: str | None = None


class ChunkItem(BaseModel):
    id: UUID
    document_id: int
    hash: str
    text: str
    meta: Optional[dict | list]
    embedding: Optional[list[float]]
    relations: Optional[dict | list]
    source_uri: Optional[str]
    index_status: Optional[KgIndexStatus]
    index_result: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class RetrievalRequest(BaseModel):
    query: str
    chat_engine: str = "default"
    top_k: Optional[int] = 5