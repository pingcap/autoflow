from typing import Optional, List

from sqlmodel import (
    Field,
    Column,
    JSON,
    Relationship as SQLRelationship,
)

from .base import UpdatableBaseModel


class RecommendQuestion(UpdatableBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    questions: List = Field(default=[], sa_column=Column(JSON))
    chat_message_id: int = Field(foreign_key="chat_messages.id", index=True)
    chat_message: "ChatMessage" = SQLRelationship(  # noqa:F821
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "RecommendQuestion.chat_message_id == ChatMessage.id",
        },
    )

    __tablename__ = "recommend_questions"
