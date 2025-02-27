from typing import Optional, Type, List
from fastapi import HTTPException
from fastapi_pagination import Params, Page
from sqlmodel import Session, SQLModel

from app.models import EntityType
from app.rag.indices.knowledge_graph.schema import (
    EntityCreate,
    SynopsisEntityCreate,
    EntityUpdate,
    EntityFilters,
    RelationshipCreate,
    RelationshipUpdate,
    RelationshipFilters,
)
from app.rag.indices.knowledge_graph.graph_store import TiDBGraphStore
from app.rag.retrievers.knowledge_graph.schema import RetrievedKnowledgeGraph
from app.staff_action import create_staff_action_log


class TiDBGraphEditor:
    def __init__(
        self,
        db_session: Session,
        graph_store: TiDBGraphStore,
    ):
        self._db_session = db_session
        self._graph_store = graph_store

    # Entities.

    def query_entities(
        self,
        filters: Optional[EntityFilters] = EntityFilters(),
        params: Params = Params(),
    ) -> Page[SQLModel]:
        return self._graph_store.fetch_entities_page(filters, params)

    def create_entity(self, create: EntityCreate) -> SQLModel:
        entity = self._graph_store.create_entity(create, commit=True)
        create_staff_action_log(
            self._db_session,
            "create_original_entity",
            "entity",
            entity.id,
            {},
            entity.screenshot(),
            commit=True,
        )
        return entity

    def create_synopsis_entity(self, create: SynopsisEntityCreate) -> SQLModel:
        synopsis_entity = self._graph_store.create_entity(create, commit=False)

        # Create relationships between synopsis entity and related entities.
        related_entities = self._graph_store.list_entities(
            EntityFilters(entity_ids=create.entities)
        )
        for related_entity in related_entities:
            self._graph_store.create_relationship(
                source_entity=synopsis_entity,
                target_entity=related_entity,
                description=f"{related_entity.name} is a part of synopsis entity (name={synopsis_entity.name}, topic={create.topic})",
                metadata={"relationship_type": EntityType.synopsis.value},
                commit=False,
            )
        self._db_session.commit()

        create_staff_action_log(
            self._db_session,
            "create_synopsis_entity",
            "entity",
            synopsis_entity.id,
            {},
            synopsis_entity.screenshot(),
            commit=True,
        )
        return synopsis_entity

    def must_get_entity(self, entity_id: int) -> Optional[Type[SQLModel]]:
        entity = self._graph_store.get_entity_by_id(entity_id)
        if entity is None:
            raise HTTPException(
                status_code=404, detail=f"Entity #{entity_id} is not found"
            )
        return entity

    def update_entity(self, entity_id: int, update: EntityUpdate) -> Type[SQLModel]:
        old_entity = self.must_get_entity(entity_id)
        old_entity_dict = old_entity.screenshot()
        new_entity = self._graph_store.update_entity(old_entity, update, commit=True)
        new_entity_dict = new_entity.screenshot()
        create_staff_action_log(
            self._db_session,
            "update",
            "entity",
            entity_id,
            old_entity_dict,
            new_entity_dict,
        )
        return new_entity

    def delete_entity(self, entity_id: int) -> Optional[Type[SQLModel]]:
        old_entity = self.must_get_entity(entity_id)
        old_entity_dict = old_entity.screenshot()
        self._graph_store.delete_entity(old_entity, commit=True)
        create_staff_action_log(
            self._db_session,
            "delete",
            "entity",
            entity_id,
            old_entity_dict,
            {},
            commit=True,
        )
        return old_entity

    def get_entity_subgraph(self, entity_id: int) -> RetrievedKnowledgeGraph:
        entity = self.must_get_entity(entity_id)
        return self._graph_store.list_relationships_by_connected_entity(entity.id)

    # Relationships.

    def must_get_relationship(self, relationship_id: int) -> Optional[Type[SQLModel]]:
        entity = self._graph_store.get_relationship_by_id(relationship_id)
        if entity is None:
            raise HTTPException(
                status_code=404, detail=f"Relationship #{relationship_id} not found"
            )
        return entity

    def create_relationship(self, create: RelationshipCreate) -> SQLModel:
        source_entity = self.must_get_entity(create.source_entity_id)
        target_entity = self.must_get_entity(create.target_entity_id)
        new_relationship = self._graph_store.create_relationship(
            source_entity=source_entity,
            target_entity=target_entity,
            description=create.description,
            metadata=create.metadata,
            commit=True,
        )
        new_relationship_dict = new_relationship.screenshot()
        create_staff_action_log(
            self._db_session,
            "create",
            "relationship",
            new_relationship.id,
            {},
            new_relationship_dict,
            commit=True,
        )
        return new_relationship

    def update_relationship(
        self, relationship_id: int, update: RelationshipUpdate
    ) -> Type[SQLModel]:
        old_relationship = self.must_get_relationship(relationship_id)
        old_relationship_dict = old_relationship.screenshot()
        new_relationship = self._graph_store.update_relationship(
            old_relationship, update, commit=True
        )
        new_relationship_dict = new_relationship.screenshot()
        create_staff_action_log(
            self._db_session,
            "update",
            "relationship",
            old_relationship.id,
            old_relationship_dict,
            new_relationship_dict,
        )
        return new_relationship

    def query_relationships(
        self,
        filters: Optional[RelationshipFilters] = RelationshipFilters(),
        params: Params = Params(),
    ) -> Page[Type[SQLModel]]:
        return self._graph_store.fetch_relationships_page(filters, params)

    def delete_relationship(self, relationship_id: int) -> None:
        relationship = self.must_get_relationship(relationship_id)
        self._graph_store.delete_relationship(relationship, commit=True)

    def batch_get_chunks_by_relationships(
        self, relationship_ids: List[int]
    ) -> List[Type[SQLModel]]:
        return self._graph_store.batch_get_chunks_by_relationships(relationship_ids)
