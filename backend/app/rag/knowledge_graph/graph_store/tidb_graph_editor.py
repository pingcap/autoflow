from typing import Optional, Tuple, List, Type

from sqlmodel import Session, select, SQLModel
from sqlalchemy.orm import joinedload
from sqlalchemy.orm.attributes import flag_modified

from app.models import Relationship, Entity, EntityType
from app.rag.knowledge_graph.graph_store.helpers import (
    get_entity_description_embedding,
    get_entity_metadata_embedding,
    get_relationship_description_embedding,
    get_query_embedding,
)

from app.rag.knowledge_graph.graph_store.tidb_graph_store import TiDBGraphStore
from app.rag.knowledge_graph.schema import Relationship as RelationshipAIModel
from app.rag.chat_config import get_default_embedding_model
from app.staff_action import create_staff_action_log


class TiDBGraphEditor:
    _entity_db_model: Type[SQLModel]
    _relationship_db_model: Type[SQLModel]


    def __init__(self, entity_model: Type[SQLModel], relationship_model: Type[SQLModel]):
        self._entity_db_model = entity_model
        self._relationship_db_model = relationship_model


    def get_entity(self, session: Session, entity_id: int) -> Optional[SQLModel]:
        return session.get(self._entity_db_model, entity_id)


    def update_entity(self, session: Session, entity: SQLModel, new_entity: dict) -> SQLModel:
        old_entity_dict = entity.screenshot()
        for key, value in new_entity.items():
            if value is not None:
                setattr(entity, key, value)
                flag_modified(entity, key)
        embed_model = get_default_embedding_model(session)
        entity.description_vec = get_entity_description_embedding(
            entity.name, entity.description, embed_model
        )
        entity.meta_vec = get_entity_metadata_embedding(entity.meta, embed_model)
        for relationship in session.exec(
            select(self._relationship_db_model)
            .options(
                joinedload(self._relationship_db_model.source_entity),
                joinedload(self._relationship_db_model.target_entity),
            )
            .where(
                (self._relationship_db_model.source_entity_id == entity.id)
                | (self._relationship_db_model.target_entity_id == entity.id)
            )
        ):
            relationship.description_vec = get_relationship_description_embedding(
                relationship.source_entity.name,
                relationship.source_entity.description,
                relationship.target_entity.name,
                relationship.target_entity.description,
                relationship.description,
                embed_model,
            )
            session.add(relationship)
        session.commit()
        session.refresh(entity)
        new_entity_dict = entity.screenshot()
        create_staff_action_log(
            session, "update", "entity", entity.id, old_entity_dict, new_entity_dict
        )
        return entity


    def get_entity_subgraph(self, session: Session, entity: SQLModel) -> Tuple[list, list]:
        """
        Get the subgraph of an entity, including all related relationships and entities.
        """
        relationships_queryset = session.exec(
            select(self._relationship_db_model)
            .options(
                joinedload(self._relationship_db_model.source_entity),
                joinedload(self._relationship_db_model.target_entity),
            )
            .where(
                (self._relationship_db_model.source_entity_id == entity.id)
                | (self._relationship_db_model.target_entity_id == entity.id)
            )
        )
        relationships = []
        entities = []
        entities_set = set()
        for relationship in relationships_queryset:
            entities_set.add(relationship.source_entity)
            entities_set.add(relationship.target_entity)
            relationships.append(relationship.screenshot())

        for entity in entities_set:
            entities.append(entity.screenshot())

        return relationships, entities


    def get_relationship(self, session: Session, relationship_id: int) -> Optional[SQLModel]:
        return session.get(self._relationship_db_model, relationship_id)


    def get_relationship_by_ids(self, session: Session, ids: list[int]) -> List[SQLModel]:
        relationships_queryset = session.exec(
            select(self._relationship_db_model)
            .where(self._relationship_db_model.id.in_(ids))
            .options(
                joinedload(self._relationship_db_model.source_entity),
                joinedload(self._relationship_db_model.target_entity),
            )
        )

        relationships = []
        entities = []
        entities_set = set()
        for relationship in relationships_queryset:
            entities_set.add(relationship.source_entity)
            entities_set.add(relationship.target_entity)
            relationships.append(relationship)

        for entity in entities_set:
            entities.append(entity)

        return entities, relationships


    def update_relationship(
        self, session: Session, relationship: SQLModel, new_relationship: dict
    ) -> SQLModel:
        old_relationship_dict = relationship.screenshot()
        for key, value in new_relationship.items():
            if value is not None:
                setattr(relationship, key, value)
                flag_modified(relationship, key)
        embed_model = get_default_embedding_model(session)
        relationship.description_vec = get_relationship_description_embedding(
            relationship.source_entity.name,
            relationship.source_entity.description,
            relationship.target_entity.name,
            relationship.target_entity.description,
            relationship.description,
            embed_model,
        )
        session.commit()
        session.refresh(relationship)
        new_relationship_dict = relationship.screenshot()
        # FIXME: some error when create staff action log
        create_staff_action_log(
            session,
            "update",
            "relationship",
            relationship.id,
            old_relationship_dict,
            new_relationship_dict,
        )
        return relationship


    def search_similar_entities(self, session: Session, query: str, top_k: int = 10) -> list:
        embed_model = get_default_embedding_model(session)
        embedding = get_query_embedding(query, embed_model)
        return session.exec(
            select(self._entity_db_model)
            .where(self._entity_db_model.entity_type == EntityType.original)
            .order_by(self._entity_db_model.description_vec.cosine_distance(embedding))
            .limit(top_k)
        ).all()


    def create_synopsis_entity(
        self,
        session: Session,
        name: str,
        description: str,
        topic: str,
        meta: dict,
        related_entities_ids: List[int],
    ) -> SQLModel:
        embed_model = get_default_embedding_model(session)
        # with session.begin():
        synopsis_entity = self._entity_db_model(
            name=name,
            description=description,
            description_vec=get_entity_description_embedding(
                name, description, embed_model
            ),
            meta=meta,
            meta_vec=get_entity_metadata_embedding(meta, embed_model),
            entity_type=EntityType.synopsis,
            synopsis_info={
                "entities": related_entities_ids,
                "topic": topic,
            },
        )
        session.add(synopsis_entity)
        graph_store = TiDBGraphStore(
            dspy_lm=None,
            session=session,
            embed_model=embed_model,
            entity_db_model=self._entity_db_model,
            relationship_db_model=self._relationship_db_model
        )
        for related_entity in session.exec(
            select(self._entity_db_model).where(self._entity_db_model.id.in_(related_entities_ids))
        ).all():
            graph_store.create_relationship(
                synopsis_entity,
                related_entity,
                RelationshipAIModel(
                    source_entity=synopsis_entity.name,
                    target_entity=related_entity.name,
                    relationship_desc=f"{related_entity.name} is a part of synopsis entity (name={synopsis_entity.name}, topic={topic})",
                ),
                {"relationship_type": EntityType.synopsis.value},
                commit=False,
            )
        session.commit()
        create_staff_action_log(
            session,
            "create_synopsis_entity",
            "entity",
            synopsis_entity.id,
            {},
            synopsis_entity.screenshot(),
            commit=False,
        )
        return synopsis_entity


legacy_tidb_graph_editor = TiDBGraphEditor(Entity, Relationship)