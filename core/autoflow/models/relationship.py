from datetime import datetime
from typing import Optional, List, Dict, Type, Any
from uuid import UUID
from sqlalchemy import Column, Text, JSON, DateTime
from sqlalchemy.orm.decl_api import RegistryType
from sqlmodel import (
    SQLModel,
    Field,
    Relationship as SQLRelationship,
)
from sqlmodel.main import default_registry
from tidb_vector.sqlalchemy import VectorType

from autoflow.utils.sql_model import PatchSQLModel


def get_relationship_model(
    table_name: str,
    entity_db_model: Type[SQLModel],
    vector_dimension: int,
    registry: RegistryType = default_registry,
) -> Type[SQLModel]:
    entity_table_name = entity_db_model.__tablename__

    class Relationship(PatchSQLModel, table=True, registry=registry):
        __tablename__ = table_name
        __table_args__ = ({"extend_existing": True},)

        id: Optional[int] = Field(default=None, primary_key=True)

        description: str = Field(sa_column=Column(Text))
        description_vec: list[float] = Field(
            sa_column=Column(VectorType(vector_dimension))
        )
        source_entity_id: int = Field(foreign_key=f"{entity_table_name}.id")
        source_entity: entity_db_model = SQLRelationship(
            sa_relationship_kwargs={
                "primaryjoin": "Relationship.source_entity_id == Entity.id",
                "lazy": "joined",
            },
        )
        target_entity_id: int = Field(foreign_key=f"{entity_table_name}.id")
        target_entity: entity_db_model = SQLRelationship(
            sa_relationship_kwargs={
                "primaryjoin": "Relationship.target_entity_id == Entity.id",
                "lazy": "joined",
            },
        )

        meta: List | Dict = Field(default={}, sa_column=Column(JSON))
        weight: int = 0
        chunk_id: Optional[UUID] = Field(default=None, nullable=True)
        document_id: Optional[int] = Field(default=None, nullable=True)

        created_at: Optional[datetime] = Field(sa_column=Column(DateTime))
        updated_at: Optional[datetime] = Field(sa_column=Column(DateTime))

        def __hash__(self):
            return hash(self.id)

        def __eq__(self, other: Any) -> bool:
            return self.id == other.id

        def screenshot(self):
            obj_dict = self.model_dump(
                exclude={
                    "description_vec",
                    "source_entity",
                    "target_entity",
                }
            )
            return obj_dict

    return Relationship
