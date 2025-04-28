import enum
from functools import lru_cache
from typing import Optional, List, Dict, Type

from sqlmodel import (
    SQLModel,
    Field,
    Column,
    JSON,
    Text,
)
from pydantic import BaseModel
from tidb_vector.sqlalchemy import VectorType
from sqlalchemy import Index
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_scoped.table_naming import get_kb_vector_dims
from app.utils.namespace import format_namespace


class EntityType(str, enum.Enum):
    original = "original"
    synopsis = "synopsis"

    def __str__(self):
        return self.value


class EntityPublic(BaseModel):
    id: int
    name: str
    description: str
    meta: Dict
    entity_type: EntityType
    synopsis_info: Dict


def get_kb_entity_model(kb: KnowledgeBase) -> Type[SQLModel]:
    vector_dimension = get_kb_vector_dims(kb)
    return get_dynamic_entity_model(vector_dimension, str(kb.id))


@lru_cache(maxsize=None)
def get_dynamic_entity_model(
    vector_dimension: int,
    namespace: Optional[str] = None,
) -> Type[SQLModel]:
    namespace = format_namespace(namespace)
    entity_table_name = f"entities_{namespace}"
    entity_model_name = f"Entity_{namespace}_{vector_dimension}"

    class Entity(SQLModel):
        id: Optional[int] = Field(default=None, primary_key=True)
        name: str = Field(max_length=512)
        description: str = Field(sa_column=Column(Text))
        meta: dict = Field(default_factory=dict, sa_column=Column(JSON))
        entity_type: EntityType = EntityType.original
        synopsis_info: List | Dict | None = Field(default=None, sa_column=Column(JSON))
        description_vec: list[float] = Field(sa_type=VectorType(vector_dimension))
        meta_vec: list[float] = Field(sa_type=VectorType(vector_dimension))

        def __hash__(self):
            return hash(self.id)

        # screenshot method is used to return a dictionary representation of the object
        # that can be used for recording or debugging purposes
        def screenshot(self):
            return self.model_dump(
                exclude={
                    "description_vec",
                    "meta_vec",
                }
            )

    entity_model = type(
        entity_model_name,
        (Entity,),
        {
            "__tablename__": entity_table_name,
            "__table_args__": (
                Index("idx_entity_type", "entity_type"),
                Index("idx_entity_name", "name"),
                {"extend_existing": True},
            ),
        },
        table=True,
    )

    return entity_model
