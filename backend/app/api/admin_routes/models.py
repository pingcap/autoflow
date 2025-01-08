from uuid import UUID
from typing import Optional
from pydantic import BaseModel

from app.api.admin_routes.embedding_model.models import EmbeddingModelItem
from app.types import LLMProvider


class LLMDescriptor(BaseModel):
    id: int
    name: str
    provider: LLMProvider
    model: str
    is_default: bool


class EmbeddingModelDescriptor(EmbeddingModelItem):
    pass


class UserDescriptor(BaseModel):
    id: UUID


class KnowledgeBaseDescriptor(BaseModel):
    id: int
    name: str


class DataSourceDescriptor(BaseModel):
    id: int
    name: str


class ChatEngineDescriptor(BaseModel):
    id: int
    name: str
    is_default: bool


class RetrieveRequest(BaseModel):
    query: str
    knowledge_base_ids: list[int] = []
    document_ids: list[int] = []
    enable_reranker: bool = True
    # If enable_reranker is True, but rerank_model_id is None, use the default rerank model
    rerank_model_id: Optional[int] = None
    top_k: Optional[int] = 5
    similarity_top_k: Optional[int] = None
    oversampling_factor: Optional[int] = 5
