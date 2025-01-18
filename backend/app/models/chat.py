import enum
from uuid import UUID
from typing import Optional, Dict
from pydantic import BaseModel
from sqlalchemy.types import TypeDecorator, Integer
from datetime import datetime

from sqlmodel import (
    Field,
    Column,
    DateTime,
    JSON,
    Relationship as SQLRelationship,
)

from .base import UUIDBaseModel, UpdatableBaseModel


class ChatVisibility(int, enum.Enum):
    PRIVATE = 0
    PUBLIC = 1

# Avoid Pydantic serializer warnings:
# When fetching values from the database, SQLAlchemy provides raw integers (0 or 1), 
# which leads to warnings during serialization because Pydantic requires the ChatVisibility enum type.

# automatically handle the conversion between int and enum types
class IntEnumType(TypeDecorator):
    impl = Integer

    def __init__(self, enum_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enum_class = enum_class

    def process_bind_param(self, value, dialect):
        # enum -> int
        if isinstance(value, self.enum_class):
            return value.value
        elif value is None:
            return None
        raise ValueError(f"Invalid value for {self.enum_class}: {value}")

    def process_result_value(self, value, dialect):
        # int -> enum
        if value is not None:
            return self.enum_class(value)
        return None

class Chat(UUIDBaseModel, UpdatableBaseModel, table=True):
    title: str = Field(max_length=256)
    engine_id: int = Field(foreign_key="chat_engines.id", nullable=True)
    engine: "ChatEngine" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Chat.engine_id == ChatEngine.id",
        },
    )
    # FIXME: why fastapi_pagination return string(json) instead of dict?
    engine_options: Dict | str = Field(default={}, sa_column=Column(JSON))
    deleted_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    user_id: UUID = Field(foreign_key="users.id", nullable=True)
    user: "User" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Chat.user_id == User.id",
        },
    )
    browser_id: str = Field(max_length=50, nullable=True)
    origin: str = Field(max_length=256, default=None, nullable=True)
    visibility: ChatVisibility = Field(
        # sa_column=Column(SmallInteger, default=ChatVisibility.PRIVATE, nullable=False)
        sa_column=Column(IntEnumType(ChatVisibility), nullable=False, default=ChatVisibility.PRIVATE)
    )

    __tablename__ = "chats"


class ChatUpdate(BaseModel):
    title: Optional[str] = None
    visibility: Optional[ChatVisibility] = None
