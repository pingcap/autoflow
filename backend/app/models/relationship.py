from datetime import datetime
from functools import lru_cache
from typing import Optional, List, Dict, Type
from uuid import UUID

from sqlalchemy import Column, Text, JSON, DateTime
from sqlmodel import (
    SQLModel,
    Field,
    Relationship as SQLRelationship,
)
from tidb_vector.sqlalchemy import VectorType
from app.utils.namespace import format_namespace


@lru_cache(maxsize=None)
def get_dynamic_relationship_model(
    vector_dimension: int,
    namespace: Optional[str] = None,
    entity_model: Optional[Type[SQLModel]] = None,
) -> Type[SQLModel]:
    namespace = format_namespace(namespace)
    entity_table_name = entity_model.__tablename__
    entity_model_name = entity_model.__name__
    relationship_table_name = f"relationships_{namespace}"
    relationship_model_name = f"Relationship_{namespace}_{vector_dimension}"

    class Relationship(SQLModel):
        id: Optional[int] = Field(default=None, primary_key=True)
        description: str = Field(sa_column=Column(Text))
        meta: List | Dict = Field(default={}, sa_column=Column(JSON))
        weight: int = 0
        source_entity_id: int = Field(foreign_key=f"{entity_table_name}.id")
        target_entity_id: int = Field(foreign_key=f"{entity_table_name}.id")
        last_modified_at: Optional[datetime] = Field(sa_column=Column(DateTime))
        document_id: Optional[int] = Field(default=None, nullable=True)
        chunk_id: Optional[UUID] = Field(default=None, nullable=True)
        description_vec: list[float] = Field(sa_type=VectorType(vector_dimension))

        def __hash__(self):
            return hash(self.id)

        def screenshot(self):
            obj_dict = self.model_dump(
                exclude={
                    "description_vec",
                    "source_entity",
                    "target_entity",
                    "last_modified_at",
                }
            )
            return obj_dict

    relationship_model = type(
        relationship_model_name,
        (Relationship,),
        {
            "__tablename__": relationship_table_name,
            "__table_args__": {"extend_existing": True},
            "__annotations__": {
                "source_entity": entity_model,
                "target_entity": entity_model,
            },
            "source_entity": SQLRelationship(
                sa_relationship_kwargs={
                    "primaryjoin": f"{relationship_model_name}.source_entity_id == {entity_model_name}.id",
                    "lazy": "joined",
                },
            ),
            "target_entity": SQLRelationship(
                sa_relationship_kwargs={
                    "primaryjoin": f"{relationship_model_name}.target_entity_id == {entity_model_name}.id",
                    "lazy": "joined",
                },
            ),
        },
        table=True,
    )

    return relationship_model
