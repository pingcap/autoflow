from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field
from autoflow.storage.graph_store.types import (
    Entity,
    EntityCreate,
    KnowledgeGraphCreate,
    Relationship,
    RelationshipCreate,
)


# Generated Knowledge Graph


class GeneratedEntity(BaseModel):
    name: str
    description: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class GeneratedRelationship(BaseModel):
    source_entity_name: str
    target_entity_name: str
    description: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class GeneratedKnowledgeGraph(BaseModel):
    entities: List[GeneratedEntity]
    relationships: List[GeneratedRelationship]

    def to_create(
        self,
        chunk_id: Optional[UUID] = None,
        document_id: Optional[UUID] = None,
    ) -> KnowledgeGraphCreate:
        return KnowledgeGraphCreate(
            entities=[
                EntityCreate(
                    name=e.name,
                    description=e.description,
                    meta=e.meta,
                )
                for e in self.entities
            ],
            relationships=[
                RelationshipCreate(
                    source_entity_name=r.source_entity_name,
                    target_entity_name=r.target_entity_name,
                    description=r.description,
                    meta=r.meta,
                    weight=0,
                    chunk_id=chunk_id,
                    document_id=document_id,
                )
                for r in self.relationships
            ],
        )


# Retrieved Knowledge Graph


class RetrievedEntity(Entity):
    similarity_score: Optional[float] = Field(default=None)
    score: Optional[float] = Field(default=None)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: "RetrievedEntity"):
        return self.id == other.id


class RetrievedRelationship(Relationship):
    similarity_score: Optional[float] = Field(default=None)
    score: Optional[float] = Field(default=None)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: "RetrievedRelationship"):
        return self.id == other.id


class RetrievedKnowledgeGraph(BaseModel):
    query: Optional[str] = Field(
        description="The query used to retrieve the knowledge graph",
        default=None,
    )
    entities: List[Entity] = Field(
        description="List of entities in the knowledge graph", default_factory=list
    )
    relationships: List[RetrievedRelationship] = Field(
        description="List of relationships in the knowledge graph", default_factory=list
    )
