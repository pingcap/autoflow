from uuid import UUID
from datetime import datetime
from typing import Optional

from sqlmodel import (
    Field,
    Column,
    JSON,
    DateTime,
    Relationship as SQLRelationship,
)

from app.models.base import UpdatableBaseModel


class BaseApiKey(UpdatableBaseModel):
    description: str = Field(max_length=100)
    api_key_display: str = Field(max_length=100)
    is_active: bool = True
    expires_at: Optional[datetime] = Field(default=None, sa_type=DateTime)
    scopes: Optional[list[str]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    knowledge_base_ids: Optional[list[int]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    chat_engine_ids: Optional[list[int]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    user_id: UUID = Field(foreign_key="users.id", nullable=False)


class ApiKey(BaseApiKey, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_secret: str = Field(max_length=255, unique=True)
    auth_version: int = Field(
        default=0, nullable=False, sa_column_kwargs={"server_default": "0"}
    )
    user: "User" = SQLRelationship(  # noqa:F821
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "ApiKey.user_id == User.id",
        },
    )

    __tablename__ = "api_keys"


class PublicApiKey(BaseApiKey):
    id: int
    user_id: UUID
