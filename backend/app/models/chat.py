import enum
from uuid import UUID
from typing import Optional, Dict
from pydantic import BaseModel
from datetime import datetime

from sqlmodel import (
    Field,
    Column,
    DateTime,
    JSON,
    SmallInteger,
    Relationship as SQLRelationship,
)

from .base import UUIDBaseModel, UpdatableBaseModel


class ChatVisibility(int, enum.Enum):
    PRIVATE = 0
    PUBLIC = 1


class Chat(UUIDBaseModel, UpdatableBaseModel, table=True):
    title: str = Field(max_length=256)
    engine_id: int = Field(foreign_key="chat_engines.id", nullable=True)
    engine: "ChatEngine" = SQLRelationship(  # noqa:F821
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Chat.engine_id == ChatEngine.id",
        },
    )
    # FIXME: why fastapi_pagination return string(json) instead of dict?
    engine_options: Dict | str = Field(default={}, sa_column=Column(JSON))
    deleted_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    user_id: UUID = Field(foreign_key="users.id", nullable=True)
    user: "User" = SQLRelationship(  # noqa:F821
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Chat.user_id == User.id",
        },
    )
    browser_id: str = Field(max_length=50, nullable=True)
    origin: str = Field(max_length=256, default=None, nullable=True)
    visibility: ChatVisibility = Field(
        sa_column=Column(SmallInteger, default=ChatVisibility.PRIVATE, nullable=False)
    )

    __tablename__ = "chats"


class ChatUpdate(BaseModel):
    title: Optional[str] = None
    visibility: Optional[ChatVisibility] = None
