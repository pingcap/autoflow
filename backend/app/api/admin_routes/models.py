from uuid import UUID
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
