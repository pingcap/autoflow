from enum import Enum
from typing import Optional, Dict
from datetime import datetime

from sqlalchemy import func
from sqlmodel import (
    Column,
    Field,
    JSON,
    DateTime,
    SQLModel,
)


class DataSourceKind(str, Enum):
    FILE = "file"
    WEB_SITEMAP = "web_sitemap"
    WEB_SINGLE_PAGE = "web_single_page"


class DataSource(SQLModel, table=True):
    __tablename__ = "data_sources"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=256)
    kind: str = Field(max_length=256)
    config: Optional[Dict] = Field(default={}, sa_column=Column(JSON))
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
        ),
    )
    deleted_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
