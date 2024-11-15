from typing import Optional, Any

from sqlmodel import Field, Column, JSON, String, SQLModel

from .base import UpdatableBaseModel, AESEncryptedColumn
from app.types import EmbeddingProvider


DEFAULT_VECTOR_DIMENSION = 1536


class EmbeddingModel(UpdatableBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=64)
    provider: EmbeddingProvider = Field(sa_column=Column(String(32), nullable=False))
    model: str = Field(max_length=256)
    vector_dimension: int = Field(default=DEFAULT_VECTOR_DIMENSION)
    config: dict | list | None = Field(sa_column=Column(JSON), default={})
    credentials: Any = Field(sa_column=Column(AESEncryptedColumn, nullable=True))
    is_default: bool = Field(default=False)

    __tablename__ = "embedding_models"
