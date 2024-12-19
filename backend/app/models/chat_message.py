from uuid import UUID
from typing import Optional, List
from datetime import datetime

from sqlmodel import (
    Field,
    Column,
    DateTime,
    Text,
    JSON,
    Relationship as SQLRelationship,
    Boolean,
    Index,
)

from .base import UpdatableBaseModel


class ChatMessage(UpdatableBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ordinal: int = Field(default=0)
    role: str = Field(max_length=64)
    content: str = Field(sa_column=Column(Text))
    error: Optional[str] = Field(sa_column=Column(Text))
    sources: List = Field(default=[], sa_column=Column(JSON))
    graph_data: dict = Field(default={}, sa_column=Column(JSON))
    meta: dict = Field(default={}, sa_column=Column(JSON))
    trace_url: Optional[str] = Field(max_length=512)
    is_best_answer: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, default=False, server_default="0"),
    )
    finished_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    chat_id: UUID = Field(foreign_key="chats.id")
    chat: "Chat" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "ChatMessage.chat_id == Chat.id",
        },
    )
    user_id: UUID = Field(foreign_key="users.id", nullable=True)
    user: "User" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "ChatMessage.user_id == User.id",
        },
    )
    post_verification_result_url: Optional[str] = Field(
        max_length=512,
        nullable=True,
    )

    __tablename__ = "chat_messages"
    __table_args__ = (Index("ix_chat_message_is_best_answer", "is_best_answer"),)
