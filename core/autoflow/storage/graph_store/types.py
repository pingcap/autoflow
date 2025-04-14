from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, model_validator, Field


# Entity


class EntityType(str, Enum):
    original = "original"
    synopsis = "synopsis"

    def __str__(self):
        return self.value


class Entity(BaseModel):
    id: Optional[UUID]
    entity_type: Optional[EntityType] = Field(
        description="Type of the entity", default=EntityType.original
    )
    name: str = Field(description="Name of the entity")
    description: str = Field(description="Description of the entity")
    embedding: Optional[list[float]] = Field(
        description="Embedding of the entity", default=None
    )
    meta: Optional[Dict[str, Any]] = Field(
        description="Metadata of the entity", default_factory=dict
    )
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class EntityCreate(BaseModel):
    entity_type: EntityType = EntityType.original
    name: Optional[str] = None
    description: Optional[str] = None
    meta: Optional[dict] = None


class SynopsisEntityCreate(EntityCreate):
    topic: str
    entities: List[int] = Field(description="The id list of the related entities")

    @model_validator(mode="after")
    def validate_entities(self):
        if len(self.entities) == 0:
            raise ValueError("Entities list should not be empty")
        return self


class EntityFilters(BaseModel):
    entity_type: Optional[EntityType] = None
    entity_id: Optional[UUID | List[UUID]] = None


class EntityUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    meta: Optional[dict] = None
    embedding: Optional[list[float]] = None


class EntityDegree(BaseModel):
    out_degree: int = 0
    in_degree: int = 0
    degrees: int = 0


# Relationship


class Relationship(BaseModel):
    id: Optional[UUID]
    source_entity_id: Optional[UUID] = Field(default=None)
    source_entity: Optional[Entity] = Field(default=None)
    target_entity_id: Optional[UUID] = Field(default=None)
    target_entity: Optional[Entity] = Field(default=None)
    description: str = Field(description="Description of the relationship")
    meta: Optional[Dict[str, Any]] = Field(
        description="Metadata of the relationship", default_factory=dict
    )
    embedding: Optional[list[float]] = Field(
        description="Embedding of the relationship", default=None
    )
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)


class RelationshipUpdate(BaseModel):
    description: Optional[str] = None
    embedding: Optional[list[float]] = None


class RelationshipFilters(BaseModel):
    entity_id: Optional[UUID | List[UUID]] = Field(
        description="Filter by the entity connected to the relationship",
        default=None,
    )
    target_entity_id: Optional[UUID | List[UUID]] = Field(
        description="Filter by the target entity of the relationship",
        default=None,
    )
    source_entity_id: Optional[UUID | List[UUID]] = Field(
        description="Filter by the source entity of the relationship",
        default=None,
    )
    chunk_id: Optional[UUID | List[UUID]] = Field(
        description="Filter by the chunk which the relationship belongs to",
        default=None,
    )
    document_id: Optional[UUID | List[UUID]] = Field(
        description="Filter by the document which the relationship belongs to",
        default=None,
    )
    relationship_id: Optional[UUID | List[UUID]] = Field(
        description="Filter by the id of the relationship",
        default=None,
    )
    metadata: Optional[Dict[str, Any]] = Field(
        description="Filter by the metadata of the relationship",
        default=None,
    )


# Knowledge Graph


class KnowledgeGraph(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
