import enum
from uuid import UUID
from typing import Optional, List

from sqlalchemy import Text, JSON

from sqlmodel import (
    Field,
    Column,
    String,
    Relationship as SQLRelationship,
)

from app.models.base import UpdatableBaseModel


class EvaluationStatus(str, enum.Enum):
    NOT_START = "not_start"
    EVALUATING = "evaluating"
    DONE = "done"
    ERROR = "error"
    CANCEL = "cancel"


class EvaluationTask(UpdatableBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=255)

    user_id: UUID = Field(foreign_key="users.id", nullable=True)
    user: "User" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "EvaluationTask.user_id == User.id",
        },
    )

    dataset_id: int = Field(nullable=True)

    evaluation_task_items: List["EvaluationTaskItem"] = SQLRelationship(
        back_populates="evaluation_task"
    )

    __tablename__ = "evaluation_tasks"


class EvaluationTaskItem(UpdatableBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chat_engine: str = Field(max_length=255)
    status: EvaluationStatus = Field(sa_column=Column(String(32), nullable=False))
    query: str = Field(sa_column=Column(Text))
    reference: str = Field(sa_column=Column(Text))
    response: str = Field(sa_column=Column(Text))
    retrieved_contexts: list[str] = Field(default=[], sa_column=Column(JSON))
    extra: dict = Field(default={}, sa_column=Column(JSON))
    error_msg: str = Field(sa_column=Column(Text, nullable=True))
    factual_correctness: Optional[float] = Field(nullable=True)
    semantic_similarity: Optional[float] = Field(nullable=True)

    evaluation_task_id: int = Field(foreign_key="evaluation_tasks.id", nullable=True)
    evaluation_task: "EvaluationTask" = SQLRelationship(
        back_populates="evaluation_task_items",
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "EvaluationTaskItem.evaluation_task_id == EvaluationTask.id",
        },
    )
    __tablename__ = "evaluation_task_items"
