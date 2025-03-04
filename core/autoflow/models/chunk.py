import enum
from datetime import datetime
from typing import Optional, Type
from uuid import UUID

from sqlalchemy import Column, DateTime, JSON, Text, func
from sqlalchemy.orm.decl_api import RegistryType
from sqlmodel import Field, SQLModel
from sqlmodel.main import default_registry, Relationship as SQLRelationship
from tidb_vector.sqlalchemy import VectorType

from autoflow.utils.sql_model import PatchSQLModel
from autoflow.utils.uuid6 import uuid7


class KgIndexStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def get_chunk_model(
    table_name: str,
    vector_dimension: int,
    document_table_name: str,
    document_db_model: Type[SQLModel],
    registry: RegistryType = default_registry,
) -> Type[SQLModel]:
    class Chunk(PatchSQLModel, table=True, registry=registry):
        __tablename__ = table_name
        __table_args__ = {"extend_existing": True}

        id: UUID = Field(primary_key=True, default_factory=uuid7)
        hash: str = Field(max_length=64)
        text: str = Field(sa_column=Column(Text))
        text_vec: list[float] = Field(sa_column=Column(VectorType(vector_dimension)))
        meta: dict = Field(default={}, sa_column=Column(JSON))
        document_id: int = Field(foreign_key=f"{document_table_name}.id", nullable=True)
        document: document_db_model = SQLRelationship()

        kg_index_status: KgIndexStatus = KgIndexStatus.NOT_STARTED
        kg_index_result: str = Field(sa_column=Column(Text, nullable=True))

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

    return Chunk
