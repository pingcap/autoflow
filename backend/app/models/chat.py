import enum
from uuid import UUID
from typing import Optional, Dict
from pydantic import BaseModel, field_serializer
from datetime import datetime

from sqlmodel import (
    Field,
    Column,
    DateTime,
    JSON,
    Relationship as SQLRelationship,
)

from .base import IntEnumType, UUIDBaseModel, UpdatableBaseModel
from app.utils.public_chat import public_engine_options, chat_hides_sources


class ChatVisibility(int, enum.Enum):
    PRIVATE = 0
    PUBLIC = 1


class Chat(UUIDBaseModel, UpdatableBaseModel, table=True):
    title: str = Field(max_length=256)
    api_key_id: Optional[int] = Field(default=None)
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
        sa_column=Column(
            IntEnumType(ChatVisibility),
            nullable=False,
            default=ChatVisibility.PRIVATE,
        )
    )

    __tablename__ = "chats"

    @field_serializer("engine_options", when_used="json")
    def public_options(self, value):
        public = public_engine_options(value)
        if chat_hides_sources(self):
            public["hide_sources"] = True
            public["knowledge_graph"]["enabled"] = False
        return public

    @field_serializer("browser_id", "api_key_id", when_used="json")
    def private_browser_id(self, value) -> None:
        return None


class ChatUpdate(BaseModel):
    title: Optional[str] = None
    visibility: Optional[ChatVisibility] = None


class ChatFilters(BaseModel):
    created_at_start: Optional[datetime] = None
    created_at_end: Optional[datetime] = None
    updated_at_start: Optional[datetime] = None
    updated_at_end: Optional[datetime] = None
    chat_origin: Optional[str] = None
    # user_id: Optional[UUID] = None          # no use now
    engine_id: Optional[int] = None


class ChatOrigin(BaseModel):
    origin: str
    chats: int
