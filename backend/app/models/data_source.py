from enum import Enum
from uuid import UUID
from typing import Optional
from datetime import datetime

from sqlmodel import (
    Column,
    Field,
    JSON,
    DateTime,
    Relationship as SQLRelationship,
)

from app.models.auth import User
from app.models.base import UpdatableBaseModel
from app.models.llm import LLM


class DataSourceType(str, Enum):
    FILE = "file"
    WEB_SITEMAP = "web_sitemap"
    WEB_SINGLE_PAGE = "web_single_page"


class DataSource(UpdatableBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=256)
    description: str = Field(max_length=512)
    data_source_type: str = Field(max_length=256)
    config: dict | list = Field(default={}, sa_column=Column(JSON))
    user_id: UUID = Field(foreign_key="users.id", nullable=True)
    user: "User" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "DataSource.user_id == User.id",
        },
    )
    deleted_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime),
    )

    # Deprecated columns.
    build_kg_index: bool = Field(default=False)
    llm_id: Optional[int] = Field(foreign_key="llms.id", nullable=True)
    llm: "LLM" = SQLRelationship(
        sa_relationship_kwargs={
            "foreign_keys": "DataSource.llm_id",
        },
    )

    __tablename__ = "data_sources"
