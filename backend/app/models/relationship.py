from datetime import datetime
from typing import Optional, Any, List, Dict
from uuid import UUID

from sqlalchemy import Column, Text, JSON, DateTime
from sqlmodel import (
    SQLModel,
    Field,
    Relationship as SQLModelRelationship,
    Relationship as SQLRelationship,
)
from tidb_vector.sqlalchemy import VectorType

from app.core.config import settings
from app.models.knowledge_base import KnowledgeBase
from app.models.entity import get_kb_entity_model, Entity
from app.models.patch.sql_model import SQLModel as PatchSQLModel
from app.models.knowledge_base_scoped.registry import get_kb_scoped_registry
from app.models.knowledge_base_scoped.table_naming import get_kb_relationships_table_name, get_kb_entities_table_name, \
    get_kb_vector_dims


class RelationshipBase(SQLModel):
    description: str = Field(sa_column=Column(Text))
    meta: List | Dict = Field(default={}, sa_column=Column(JSON))
    weight: int = 0
    source_entity_id: int = Field(foreign_key="entities.id")
    target_entity_id: int = Field(foreign_key="entities.id")
    last_modified_at: Optional[datetime] = Field(sa_column=Column(DateTime))
    document_id: Optional[int] = Field(default=None, nullable=True)
    chunk_id: Optional[UUID] = Field(default=None, nullable=True)


# Notice: DO NOT forget to modify the definition in `get_kb_chunk_model` to
# keep the table structure on both sides consistent.
class Relationship(RelationshipBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    description_vec: Any = Field(
        sa_column=Column(
            VectorType(settings.EMBEDDING_DIMS), comment="hnsw(distance=cosine)"
        )
    )
    source_entity: Entity = SQLModelRelationship(
        sa_relationship_kwargs={
            "primaryjoin": "Relationship.source_entity_id == Entity.id",
            "lazy": "joined",
        },
    )
    target_entity: Entity = SQLModelRelationship(
        sa_relationship_kwargs={
            "primaryjoin": "Relationship.target_entity_id == Entity.id",
            "lazy": "joined",
        },
    )

    __tablename__ = "relationships"

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


class RelationshipPublic(RelationshipBase):
    id: int


def get_kb_relationship_model(kb: KnowledgeBase) -> SQLModel:
    vector_dimension = get_kb_vector_dims(kb)
    entities_table_name = get_kb_entities_table_name(kb)
    relationships_table_name = get_kb_relationships_table_name(kb)
    ctx = get_kb_scoped_registry(kb)

    if ctx.entity_model:
        entity_model = ctx.entity_model
    else:
        entity_model = get_kb_entity_model(kb)

    if ctx.relationship_model:
        return ctx.relationship_model

    class KBRelationship(PatchSQLModel, table=True, registry=ctx.registry):
        __tablename__ = relationships_table_name
        __table_args__ = ({'extend_existing': True},)

        id: Optional[int] = Field(default=None, primary_key=True)
        description: str = Field(sa_column=Column(Text))
        meta: List | Dict = Field(default={}, sa_column=Column(JSON))
        weight: int = 0
        source_entity_id: int = Field(foreign_key=f"{entities_table_name}.id")
        source_entity: entity_model = SQLRelationship(
            sa_relationship_kwargs={
                "primaryjoin": f"KBRelationship.source_entity_id == KBEntity.id",
                "lazy": "joined",
            },
        )
        target_entity_id: int = Field(foreign_key=f"{entities_table_name}.id")
        target_entity: entity_model = SQLRelationship(
            sa_relationship_kwargs={
                "primaryjoin": f"KBRelationship.target_entity_id == KBEntity.id",
                "lazy": "joined",
            },
        )
        last_modified_at: Optional[datetime] = Field(sa_column=Column(DateTime))
        document_id: Optional[int] = Field(default=None, nullable=True)
        chunk_id: Optional[UUID] = Field(default=None, nullable=True)
        description_vec: list[float] = Field(
            sa_column=Column(
                VectorType(vector_dimension), comment="hnsw(distance=cosine)"
            )
        )

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

    ctx.relationship_model = KBRelationship
    return KBRelationship
