from typing import Optional, Any

from sqlmodel import Field, Column, JSON, String

from .base import UpdatableBaseModel, AESEncryptedColumn
from app.rag.rerankers.provider import RerankerProvider
from pydantic import BaseModel


class BaseRerankerModel(UpdatableBaseModel):
    name: str = Field(max_length=64)
    provider: RerankerProvider = Field(sa_column=Column(String(32), nullable=False))
    model: str = Field(max_length=256)
    top_n: int = Field(default=10)
    config: dict | list | None = Field(sa_column=Column(JSON), default={})
    is_default: bool = Field(default=False)


class RerankerModel(BaseRerankerModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    credentials: Any = Field(sa_column=Column(AESEncryptedColumn, nullable=True))

    __tablename__ = "reranker_models"


class AdminRerankerModel(BaseRerankerModel):
    id: int


class RerankerModelUpdate(BaseModel):
    name: Optional[str] = None
    config: Optional[dict | list] = None
    credentials: Optional[str | dict] = None
    top_n: Optional[int] = None
