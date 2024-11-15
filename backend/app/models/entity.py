import enum
from typing import Optional, Any, List, Dict

from sqlmodel import (
    SQLModel,
    Field,
    Column,
    JSON,
    Text,
)
from tidb_vector.sqlalchemy import VectorType
from sqlalchemy import Index

from app.core.config import settings
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_scoped.registry import get_kb_scoped_registry
from app.models.knowledge_base_scoped.table_naming import get_kb_entities_table_name, get_kb_vector_dims
from app.models.patch.sql_model import SQLModel as PatchSQLModel


class EntityType(str, enum.Enum):
    original = "original"
    synopsis = "synopsis"

    def __str__(self):
        return self.value


class EntityBase(SQLModel):
    name: str = Field(max_length=512)
    description: str = Field(sa_column=Column(Text))
    meta: List | Dict = Field(default={}, sa_column=Column(JSON))
    entity_type: EntityType = EntityType.original
    synopsis_info: List | Dict | None = Field(default=None, sa_column=Column(JSON))


# Notice: DO NOT forget to modify the definition in `get_kb_chunk_model` to
# keep the table structure on both sides consistent.
class Entity(EntityBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    description_vec: Any = Field(
        sa_column=Column(
            VectorType(settings.EMBEDDING_DIMS), comment="hnsw(distance=cosine)"
        )
    )
    meta_vec: Any = Field(
        sa_column=Column(
            VectorType(settings.EMBEDDING_DIMS), comment="hnsw(distance=cosine)"
        )
    )

    __tablename__ = "entities"
    __table_args__ = (Index("idx_entity_type", "entity_type"),)

    def __hash__(self):
        return hash(self.id)

    # screenshot method is used to return a dictionary representation of the object
    # that can be used for recording or debugging purposes
    def screenshot(self):
        return self.model_dump(exclude={"description_vec", "meta_vec"})


# Public Entity model will be used in API response
class EntityPublic(EntityBase):
    id: int


def get_kb_entity_model(kb: KnowledgeBase) -> SQLModel:
    vector_dimension = get_kb_vector_dims(kb)
    entities_table_name = get_kb_entities_table_name(kb)
    ctx = get_kb_scoped_registry(kb)

    if ctx.entity_model:
        return ctx.entity_model

    class KBEntity(PatchSQLModel, table=True, registry=ctx.registry):
        __tablename__ = entities_table_name
        __table_args__ = (
            Index("idx_entity_type", "entity_type"),
            {'extend_existing': True}
        )

        id: Optional[int] = Field(default=None, primary_key=True)
        name: str = Field(max_length=512)
        description: str = Field(sa_column=Column(Text))
        meta: List | Dict = Field(default={}, sa_column=Column(JSON))
        entity_type: EntityType = EntityType.original
        synopsis_info: List | Dict | None = Field(default=None, sa_column=Column(JSON))
        description_vec: Any = Field(
            sa_column=Column(
                VectorType(vector_dimension), comment="hnsw(distance=cosine)"
            )
        )
        meta_vec: Any = Field(
            sa_column=Column(
                VectorType(vector_dimension), comment="hnsw(distance=cosine)"
            )
        )

        def __hash__(self):
            return hash(self.id)

        def screenshot(self):
            return self.model_dump(exclude={"description_vec", "meta_vec"})

    ctx.entity_model = KBEntity
    return KBEntity


