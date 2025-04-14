from abc import ABC

import logging
from typing import (
    Collection,
    Dict,
    List,
    Optional,
    Tuple,
)
from uuid import UUID

from autoflow.storage.types import QueryBundle
from autoflow.types import BaseComponent
from autoflow.storage.graph_store.types import (
    Entity,
    EntityFilters,
    EntityType,
    EntityUpdate,
    EntityDegree,
    KnowledgeGraph,
    KnowledgeGraphCreate,
    Relationship,
    RelationshipFilters,
    RelationshipUpdate,
)


logger = logging.getLogger(__name__)


class GraphStore(BaseComponent, ABC):
    """Abstract base class for knowledge graph storage"""

    # Entity Basic Operations

    def list_entities(
        self, filters: Optional[EntityFilters] = EntityFilters()
    ) -> List[Entity]:
        """List all entities matching the filters"""
        raise NotImplementedError

    def search_entities(
        self,
        query: QueryBundle,
        top_k: int = 10,
        num_candidate: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        filters: Optional[EntityFilters] = None,
    ) -> List[Tuple[Entity, float]]:
        raise NotImplementedError

    def get_entity(self, entity_id: UUID) -> Entity:
        """Get entity by ID"""
        raise NotImplementedError

    def must_get_entity(self, entity_id: UUID) -> Entity:
        entity = self.get_entity(entity_id)
        if entity is None:
            raise ValueError(f"Can not find the entity #{entity_id}")
        return entity

    def create_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.original,
        description: Optional[str] = None,
        meta: Optional[dict] = None,
        embedding: Optional[list[float]] = None,
    ) -> Entity:
        """Create a new entity"""
        raise NotImplementedError

    def update_entity(self, entity: Entity | UUID, update: EntityUpdate) -> Entity:
        """Update an existing entity"""
        raise NotImplementedError

    def delete_entity(self, entity_id: UUID) -> None:
        """Delete an entity"""
        raise NotImplementedError

    def delete_orphan_entities(self):
        """Remove entities that have no relationships"""
        raise NotImplementedError

    # Entity Degree Operations

    def calc_entity_out_degree(self, entity_id: UUID) -> int:
        """Calculate out-degree of an entity"""
        raise NotImplementedError

    def calc_entity_in_degree(self, entity_id: UUID) -> int:
        """Calculate in-degree of an entity"""
        raise NotImplementedError

    def calc_entity_degree(self, entity_id: UUID) -> int:
        """Calculate total degree of an entity"""
        raise NotImplementedError

    def calc_entities_degrees(
        self, entity_ids: Collection[UUID]
    ) -> Dict[UUID, EntityDegree]:
        """Calculate degrees for multiple entities in bulk"""
        raise NotImplementedError

    # Relationship Basic Operations

    def get_relationship(self, relationship_id: UUID) -> Relationship:
        """Get relationship by ID"""
        raise NotImplementedError

    def list_relationships(self, filters: RelationshipFilters) -> List[Relationship]:
        """List all relationships matching the filters"""
        raise NotImplementedError

    def create_relationship(
        self,
        source_entity: Entity,
        target_entity: Entity,
        description: Optional[str] = None,
        meta: Optional[dict] = {},
        **kwargs,
    ) -> Relationship:
        """Create a new relationship between entities"""
        raise NotImplementedError

    def update_relationship(
        self, relationship: Relationship | UUID, update: RelationshipUpdate
    ) -> Relationship:
        """Update an existing relationship"""
        raise NotImplementedError

    def delete_relationship(self, relationship_id: UUID):
        """Delete a relationship"""
        raise NotImplementedError

    def search_relationships(
        self,
        query: QueryBundle,
        top_k: int = 10,
        num_candidate: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        distance_range: Optional[Tuple[float, float]] = None,
        filters: Optional[RelationshipFilters] = None,
    ) -> List[Tuple[Relationship, float]]:
        """

        Args:
            query:
            top_k:
            num_candidate:
            distance_threshold:
            distance_range:
            filters:
        """
        raise NotImplementedError

    def reset(self):
        """Reset the graph store"""
        raise NotImplementedError

    def drop(self):
        """Drop the graph store"""
        raise NotImplementedError

    # Knowledge Graph Operations

    def add(self, knowledge_graph: KnowledgeGraphCreate) -> Optional[KnowledgeGraph]:
        """Add a knowledge graph to the graph store"""
        raise NotImplementedError
