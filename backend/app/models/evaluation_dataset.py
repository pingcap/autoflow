from uuid import UUID
from typing import Optional, List

from sqlalchemy import Text, JSON

from sqlmodel import (
    Field,
    Column,
    Relationship as SQLRelationship,
)

from app.models.base import UpdatableBaseModel


class EvaluationDataset(UpdatableBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=255)

    user_id: UUID = Field(foreign_key="users.id", nullable=True)
    user: "User" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "EvaluationDataset.user_id == User.id",
        },
    )

    evaluation_data_list: List["EvaluationDatasetItem"] = SQLRelationship(
        back_populates="evaluation_dataset"
    )

    __tablename__ = "evaluation_datasets"


class EvaluationDatasetItem(UpdatableBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    query: str = Field(sa_column=Column(Text))
    reference: str = Field(sa_column=Column(Text))
    retrieved_contexts: list[str] = Field(default=[], sa_column=Column(JSON))
    extra: dict = Field(default={}, sa_column=Column(JSON))

    evaluation_dataset_id: int = Field(
        foreign_key="evaluation_datasets.id", nullable=True
    )
    evaluation_dataset: "EvaluationDataset" = SQLRelationship(
        back_populates="evaluation_data_list",
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "EvaluationDatasetItem.evaluation_dataset_id == EvaluationDataset.id",
        },
    )
    __tablename__ = "evaluation_dataset_items"
