from datetime import datetime
from functools import lru_cache
from typing import Optional, Type
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import Column, Text, JSON, DateTime
from sqlmodel import (
    SQLModel,
    Field,
    Relationship as SQLRelationship,
)
from tidb_vector.sqlalchemy import VectorType
from app.models.entity import get_kb_entity_model
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_scoped.table_naming import get_kb_vector_dims
from app.utils.namespace import format_namespace
from app.logger import logger


class RelationshipPublic(BaseModel):
    id: int
    description: str
    source_entity_id: int
    target_entity_id: int
    meta: dict = Field(default_factory=dict)
    weight: Optional[int] = Field(default=0)
    last_modified_at: Optional[datetime] = Field(default=None)
    document_id: Optional[int] = Field(default=None)
    chunk_id: Optional[UUID] = Field(default=None)


def get_kb_relationship_model(kb: KnowledgeBase) -> Type[SQLModel]:
    vector_dimension = get_kb_vector_dims(kb)
    entity_model = get_kb_entity_model(kb)
    return get_dynamic_relationship_model(vector_dimension, str(kb.id), entity_model)


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

    logger.info(
        "Dynamic create relationship model (dimension: %s, table: %s, model: %s)",
        vector_dimension,
        relationship_table_name,
        relationship_model_name,
    )

    class Relationship(SQLModel):
        id: Optional[int] = Field(default=None, primary_key=True)
        description: str = Field(sa_column=Column(Text))
        meta: dict = Field(default_factory=dict, sa_column=Column(JSON))
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
