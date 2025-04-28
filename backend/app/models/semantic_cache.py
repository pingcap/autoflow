from typing import Optional, Any
from datetime import datetime

from sqlmodel import (
    SQLModel,
    Field,
    Column,
    JSON,
    Text,
    func,
    DateTime,
)
from tidb_vector.sqlalchemy import VectorType

from app.core.config import settings


class SemanticCache(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    query: str = Field(sa_column=Column(Text))
    query_vec: Any = Field(
        sa_column=Column(
            VectorType(settings.EMBEDDING_DIMS), comment="hnsw(distance=cosine)"
        )
    )
    value: str = Field(sa_column=Column(Text))
    value_vec: Any = Field(
        sa_column=Column(
            VectorType(settings.EMBEDDING_DIMS), comment="hnsw(distance=cosine)"
        )
    )
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(
        sa_column=Column(DateTime, server_default=func.now(), nullable=True)
    )
    updated_at: datetime = Field(
        sa_column=Column(
            DateTime, server_default=func.now(), onupdate=func.now(), nullable=True
        )
    )

    __tablename__ = "semantic_cache"
    __table_args__ = {
        # Ref: https://docs.pingcap.com/tidb/stable/time-to-live
        "mysql_TTL": "created_at + INTERVAL 1 MONTH;",
    }

    def __hash__(self):
        return hash(self.id)

    # screenshot method is used to return a dictionary representation of the object
    # that can be used for recording or debugging purposes
    def screenshot(self):
        return self.model_dump(exclude={"query_vec", "value_vec"})
