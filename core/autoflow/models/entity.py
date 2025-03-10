import enum
from typing import Optional, Any, List, Dict, Type

from sqlalchemy.orm.decl_api import RegistryType
from sqlmodel import (
    Field,
    Column,
    JSON,
    Text,
)
from sqlalchemy import Index
from sqlmodel.main import default_registry, SQLModel
from tidb_vector.sqlalchemy import VectorType

from autoflow.utils.sql_model import PatchSQLModel


class EntityType(str, enum.Enum):
    original = "original"
    synopsis = "synopsis"

    def __str__(self):
        return self.value


def get_entity_model(
    table_name: str,
    vector_dimension: int,
    registry: RegistryType = default_registry,
) -> Type[SQLModel]:
    class Entity(PatchSQLModel, table=True, registry=registry):
        __tablename__ = table_name
        __table_args__ = (
            Index("idx_entity_type", "entity_type"),
            Index("idx_entity_name", "name"),
            {"extend_existing": True},
        )

        id: Optional[int] = Field(default=None, primary_key=True)
        entity_type: EntityType = EntityType.original
        name: str = Field(max_length=512)
        description: str = Field(sa_column=Column(Text))
        description_vec: list[float] = Field(
            default=None, sa_column=Column(VectorType(vector_dimension))
        )
        meta: List | Dict = Field(default={}, sa_column=Column(JSON))
        meta_vec: list[float] = Field(
            default=None, sa_column=Column(VectorType(vector_dimension))
        )
        synopsis_info: List | Dict | None = Field(default=None, sa_column=Column(JSON))

        def __hash__(self):
            return hash(self.id)

        def __eq__(self, other: Any) -> bool:
            return self.id == other.id

        def screenshot(self):
            return self.model_dump(exclude={"description_vec", "meta_vec"})

    return Entity
