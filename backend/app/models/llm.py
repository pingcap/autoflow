from typing import Optional, Any
from sqlmodel import Field, Column, JSON, String
from pydantic import BaseModel
from app.rag.llms.provider import LLMProvider
from .base import UpdatableBaseModel, AESEncryptedColumn


class BaseLLM(UpdatableBaseModel):
    name: str = Field(max_length=64)
    provider: LLMProvider = Field(sa_column=Column(String(32), nullable=False))
    model: str = Field(max_length=256)
    config: dict | list | None = Field(sa_column=Column(JSON), default={})
    is_default: bool = Field(default=False)


class LLM(BaseLLM, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    credentials: Any = Field(sa_column=Column(AESEncryptedColumn, nullable=True))

    __tablename__ = "llms"


class AdminLLM(BaseLLM):
    id: int


class LLMUpdate(BaseModel):
    name: Optional[str] = None
    config: Optional[dict] = None
    credentials: Optional[str | dict] = None
