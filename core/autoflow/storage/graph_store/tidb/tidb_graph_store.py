import json
import logging
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import tidb_vector
import sqlalchemy

from typing import List, Optional, Tuple, Dict, Type, Sequence, Any, Mapping, Collection

from deepdiff import DeepDiff
from sqlalchemy.orm import defer, joinedload
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, asc, func, select, text, or_, SQLModel
from tidb_vector.sqlalchemy import VectorAdaptor
from sqlalchemy import desc, Engine

from autoflow.indices.knowledge_graph.schema import (
    AIKnowledgeGraph,
    ProcessedAIKnowledgeGraph,
    AIRelationshipWithEntityDesc,
    AIEntity,
)
from autoflow.llms.embeddings import EmbeddingModel
from autoflow.models.entity import EntityType
from autoflow.storage.graph_store import KnowledgeGraphStore
from autoflow.storage.graph_store.base import (
    EntityFilters,
    EntityCreate,
    EntityUpdate,
    EntityDegree,
    RelationshipUpdate,
    RelationshipFilters,
    RetrievedEntity,
    RetrievedRelationship,
    E,
    R,
    C,
)
from autoflow.storage.schema import QueryBundle

logger = logging.getLogger(__name__)


def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def _process_ai_knowledge_graph(
    knowledge_graph: AIKnowledgeGraph, chunk_metadata: Mapping[str, Any]
) -> ProcessedAIKnowledgeGraph:
    entities = knowledge_graph.entities
    relationships = []
    mapped_entities = {entity.name: entity for entity in entities}

    for relationship in knowledge_graph.relationships:
        if relationship.source_entity not in mapped_entities:
            new_source_entity = AIEntity(
                name=relationship.source_entity,
                description=(
                    f"Derived from from relationship: "
                    f"{relationship.source_entity} -> {relationship.relationship_desc} -> {relationship.target_entity}"
                ),
                metadata={"status": "need-revised"},
            )
            entities.append(new_source_entity)
            mapped_entities[relationship.source_entity] = new_source_entity
            source_entity_description = new_source_entity.description
        else:
            source_entity_description = mapped_entities[
                relationship.source_entity
            ].description

        if relationship.target_entity not in mapped_entities:
            new_target_entity = AIEntity(
                name=relationship.target_entity,
                description=(
                    f"Derived from from relationship: "
                    f"{relationship.source_entity} -> {relationship.relationship_desc} -> {relationship.target_entity}"
                ),
                metadata={"status": "need-revised"},
            )
            entities.append(new_target_entity)
            mapped_entities[relationship.target_entity] = new_target_entity
            target_entity_description = new_target_entity.description
        else:
            target_entity_description = mapped_entities[
                relationship.target_entity
            ].description

        relationships.append(
            AIRelationshipWithEntityDesc(
                source_entity=relationship.source_entity,
                source_entity_description=source_entity_description,
                target_entity=relationship.target_entity,
                target_entity_description=target_entity_description,
                relationship_desc=relationship.relationship_desc,
                meta={**chunk_metadata},
            )
        )

    return ProcessedAIKnowledgeGraph(
        entities=entities,
        relationships=relationships,
    )


class TiDBKnowledgeGraphStore(KnowledgeGraphStore[E, R, C]):
    def __init__(
        self,
        db_engine: Engine,
        embedding_model: EmbeddingModel,
        entity_db_model: Type[E],
        relationship_db_model: Type[R],
        entity_similarity_threshold: Optional[float] = 0.9,
    ):
        super().__init__(embedding_model)
        self._db_engine = db_engine
        self._embedding_model = embedding_model
        self._entity_db_model = entity_db_model
        self._relationship_db_model = relationship_db_model
        self._entity_distance_threshold = 1 - entity_similarity_threshold

    # Schema Operations

    # TODO: move to low-level storage API.
    def ensure_table_schema(self) -> None:
        inspector = sqlalchemy.inspect(self._db_engine)
        existed_table_names = inspector.get_table_names()

        entity_model = self._entity_db_model
        entities_table_name = entity_model.__tablename__
        if entities_table_name not in existed_table_names:
            entity_model.metadata.create_all(
                self._db_engine, tables=[entity_model.__table__]
            )

            # Add HNSW index to accelerate ann queries.
            VectorAdaptor(self._db_engine).create_vector_index(
                entity_model.description_vec, tidb_vector.DistanceMetric.COSINE
            )
            VectorAdaptor(self._db_engine).create_vector_index(
                entity_model.meta_vec, tidb_vector.DistanceMetric.COSINE
            )

            logger.info(
                f"Entities table <{entities_table_name}> has been created successfully."
            )
        else:
            logger.info(
                f"Entities table <{entities_table_name}> is already exists, not action to do."
            )

        relationship_model = self._relationship_db_model
        relationships_table_name = relationship_model.__tablename__
        if relationships_table_name not in existed_table_names:
            relationship_model.metadata.create_all(
                self._db_engine, tables=[relationship_model.__table__]
            )

            # Add HNSW index to accelerate ann queries.
            VectorAdaptor(self._db_engine).create_vector_index(
                relationship_model.description_vec,
                tidb_vector.DistanceMetric.COSINE,
            )

            logger.info(
                f"Relationships table <{relationships_table_name}> has been created successfully."
            )
        else:
            logger.info(
                f"Relationships table <{relationships_table_name}> is already exists, not action to do."
            )

    def drop_table_schema(self) -> None:
        inspector = sqlalchemy.inspect(self._db_engine)
        existed_table_names = inspector.get_table_names()
        relationships_table_name = self._relationship_db_model.__tablename__
        entities_table_name = self._entity_db_model.__tablename__

        if relationships_table_name in existed_table_names:
            self._relationship_db_model.metadata.drop_all(
                self._db_engine, tables=[self._relationship_db_model.__table__]
            )
            logger.info(
                f"Relationships table <{relationships_table_name}> has been dropped successfully."
            )
        else:
            logger.info(
                f"Relationships table <{relationships_table_name}> is not existed, not action to do."
            )

        if entities_table_name in existed_table_names:
            self._entity_db_model.metadata.drop_all(
                self._db_engine, tables=[self._entity_db_model.__table__]
            )
            logger.info(
                f"Entities table <{entities_table_name}> has been dropped successfully."
            )
        else:
            logger.info(
                f"Entities table <{entities_table_name}> is not existed, not action to do."
            )

    @contextmanager
    def _session_scope(self, session: Optional[Session] = None, commit: bool = False):
        """Provide a transactional scope around a series of operations."""
        should_close = session is None
        session = session or Session(self._db_engine)

        try:
            yield session
            if commit:
                session.commit()
            else:
                session.flush()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if should_close:
                session.close()

    def _get_entity_description_embedding(
        self, name: str, description: str
    ) -> List[float]:
        # TODO: Make it configurable.
        embedding_str = f"{name}: {description}"
        return self._embedding_model.get_text_embedding(embedding_str)

    def _get_entity_metadata_embedding(self, metadata: Dict[str, Any]) -> List[float]:
        embedding_str = json.dumps(metadata, ensure_ascii=False)
        return self._embedding_model.get_text_embedding(embedding_str)

    def _get_relationship_description_embedding(
        self,
        source_entity_name: str,
        source_entity_description,
        target_entity_name: str,
        target_entity_description: str,
        relationship_desc: str,
    ) -> List[float]:
        # TODO: Make it configurable.
        embedding_str = (
            f"{source_entity_name}({source_entity_description}) -> "
            f"{relationship_desc} -> {target_entity_name}({target_entity_description}) "
        )
        return self._embedding_model.get_text_embedding(embedding_str)

    def save_knowledge_graph(
        self, knowledge_graph: AIKnowledgeGraph, chunk: SQLModel
    ) -> None:
        if (
            len(knowledge_graph.entities) == 0
            or len(knowledge_graph.relationships) == 0
        ):
            logger.info(
                "Entities or relationships are empty, skip saving to the database"
            )
            return

        chunk_id = chunk.id
        exists_relationships = self.list_relationships(
            RelationshipFilters(chunk_ids=[chunk_id])
        )
        if len(exists_relationships) > 0:
            logger.info(f"{chunk_id} already exists in the relationship table, skip.")
            return

        knowledge_graph = _process_ai_knowledge_graph(knowledge_graph, chunk.meta)

        with self._session_scope() as session:
            entities_name_map = defaultdict(list)
            for entity in knowledge_graph.entities:
                entities_name_map[entity.name].append(
                    self.find_or_create_entity(
                        EntityCreate(
                            entity_type=EntityType.original,
                            name=entity.name,
                            description=entity.description,
                            meta=entity.metadata,
                        ),
                        commit=False,
                        db_session=session,
                    )
                )

            def _find_or_create_entity_for_relation(
                name: str, description: str
            ) -> SQLModel:
                _embedding = self._get_entity_description_embedding(name, description)

                # Check entities_name_map first, if not found, then check the database.
                entities_with_same_name = entities_name_map.get(name, [])
                if len(entities_with_same_name) > 0:
                    the_closest_entity = None
                    the_closest_distance = float("inf")
                    for en in entities_with_same_name:
                        distance = cosine_distance(en.description_vec, _embedding)
                        if distance < the_closest_distance:
                            the_closest_distance = distance
                            the_closest_entity = en
                    if the_closest_distance < self._entity_distance_threshold:
                        return the_closest_entity

                return self.find_or_create_entity(
                    EntityCreate(
                        entity_type=EntityType.original,
                        name=name,
                        description=description,
                        meta=entity.metadata,
                    ),
                    commit=False,
                    db_session=session,
                )

            for r in knowledge_graph.relationships:
                logger.info(
                    "Save entities for relationship: %s -> %s -> %s",
                    r.source_entity,
                    r.relationship_desc,
                    r.target_entity,
                )
                source_entity = _find_or_create_entity_for_relation(
                    r.source_entity, r.source_entity_description
                )
                target_entity = _find_or_create_entity_for_relation(
                    r.target_entity, r.target_entity_description
                )

                self.create_relationship(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    description=r.relationship_desc,
                    commit=False,
                    db_session=session,
                )
            session.commit()

    # Entity Basic Operations

    def list_entities(
        self,
        filters: Optional[EntityFilters] = EntityFilters(),
        db_session: Session = None,
    ) -> Sequence[E]:
        with self._session_scope(db_session) as session:
            stmt = self._build_entities_query(filters)
            return session.exec(stmt).all()

    def _build_entities_query(self, filters: EntityFilters):
        stmt = select(self._entity_db_model)
        if filters.entity_type:
            stmt = stmt.where(self._entity_db_model.entity_type == filters.entity_type)
        if filters.search:
            stmt = stmt.where(
                or_(
                    self._entity_db_model.name.like(f"%{filters.search}%"),
                    self._entity_db_model.description.like(f"%{filters.search}%"),
                )
            )
        return stmt

    def get_entity_by_id(self, entity_id: int) -> Type[E]:
        with self._session_scope() as session:
            return session.get(self._entity_db_model, entity_id)

    def must_get_entity_by_id(self, entity_id: int) -> Type[E]:
        entity = self.get_entity_by_id(entity_id)
        if entity is None:
            raise ValueError(f"Entity <{entity_id}> does not exist")
        return entity

    def create_entity(
        self, create: EntityCreate, commit: bool = True, db_session: Session = None
    ) -> E:
        desc_vec = self._get_entity_description_embedding(
            create.name, create.description
        )
        meta_vec = self._get_entity_metadata_embedding(create.meta)
        entity = self._entity_db_model(
            name=create.name,
            entity_type=EntityType.original,
            description=create.description,
            description_vec=desc_vec,
            meta=create.meta,
            meta_vec=meta_vec,
        )

        with self._session_scope(db_session, commit) as session:
            session.add(entity)
        return entity

    def find_or_create_entity(
        self, create: EntityCreate, commit: bool = True, db_session: Session = None
    ) -> E:
        most_similar_entity = self._get_the_most_similar_entity(
            create, db_session=db_session
        )

        if most_similar_entity is not None:
            return most_similar_entity

        return self.create_entity(create, commit=commit, db_session=db_session)

    def update_entity(
        self,
        entity: Type[E],
        update: EntityUpdate,
        commit: bool = True,
        db_session: Session = None,
    ) -> Type[E]:
        for key, value in update.model_dump().items():
            if value is None:
                continue
            setattr(entity, key, value)
            flag_modified(entity, key)

        entity.description_vec = self._get_entity_description_embedding(
            entity.name, entity.description
        )
        if update.meta is not None:
            entity.meta_vec = self._get_entity_metadata_embedding(entity.meta)

        with self._session_scope(db_session, commit) as session:
            session.add(entity)
            # Update linked relationships.
            connected_relationships = self.list_entity_relationships(entity.id)
            for relationship in connected_relationships:
                self.update_relationship(relationship, RelationshipUpdate(), commit)
            session.refresh(entity)
            return entity

    def delete_entity(
        self, entity: Type[E], commit: bool = True, db_session: Session = None
    ) -> None:
        with self._session_scope(db_session, commit) as session:
            # Delete linked relationships.
            linked_relationships = self.list_entity_relationships(entity.id)
            for relationship in linked_relationships:
                session.delete(relationship)

            session.delete(entity)

    def list_entity_relationships(self, entity_id: int) -> List[R]:
        stmt = (
            select(self._relationship_db_model)
            .options(
                defer(self._relationship_db_model.description_vec),
                joinedload(self._relationship_db_model.source_entity)
                .defer(self._entity_db_model.description_vec)
                .defer(self._entity_db_model.meta_vec),
                joinedload(self._relationship_db_model.target_entity)
                .defer(self._entity_db_model.description_vec)
                .defer(self._entity_db_model.meta_vec),
            )
            .where(
                (self._relationship_db_model.source_entity_id == entity_id)
                | (self._relationship_db_model.target_entity_id == entity_id)
            )
        )
        with self._session_scope() as session:
            return session.exec(stmt).all()

    def calc_entity_out_degree(self, entity_id: int) -> Optional[int]:
        stmt = select(func.count(self._relationship_db_model.id)).where(
            self._relationship_db_model.source_entity_id == entity_id
        )
        with self._session_scope() as session:
            return session.exec(stmt).one()

    def calc_entity_in_degree(self, entity_id: int) -> Optional[int]:
        stmt = select(func.count(self._relationship_db_model.id)).where(
            self._relationship_db_model.target_entity_id == entity_id
        )
        with self._session_scope() as session:
            return session.exec(stmt).one()

    def calc_entity_degree(self, entity_id: int) -> Optional[int]:
        stmt = select(func.count(self._relationship_db_model.id)).where(
            or_(
                self._relationship_db_model.target_entity_id == entity_id,
                self._relationship_db_model.source_entity_id == entity_id,
            )
        )
        with self._session_scope() as session:
            return session.exec(stmt).one()

    def bulk_calc_entities_degrees(
        self, entity_ids: Collection[int]
    ) -> Mapping[int, EntityDegree]:
        stmt = (
            select(
                self._entity_db_model.id,
                func.count(self._relationship_db_model.id)
                .filter(
                    self._relationship_db_model.source_entity_id
                    == self._entity_db_model.id
                )
                .label("out_degree"),
                func.count(self._relationship_db_model.id)
                .filter(
                    self._relationship_db_model.target_entity_id
                    == self._entity_db_model.id
                )
                .label("in_degree"),
            )
            .where(self._entity_db_model.id.in_(entity_ids))
            .outerjoin(self._relationship_db_model)
            .group_by(self._entity_db_model.id)
        )

        with self._session_scope() as session:
            results = session.exec(stmt).all()
            return {
                item.id: EntityDegree(
                    in_degree=item.in_degree,
                    out_degree=item.out_degree,
                    degrees=item.in_degree + item.out_degree,
                )
                for item in results
            }

    # Entities Retrieve Operations

    def retrieve_entities(
        self,
        query: str,
        entity_type: EntityType = EntityType.original,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedEntity]:
        entities = self.search_similar_entities(
            query=QueryBundle(query_str=query),
            top_k=top_k,
            nprobe=nprobe,
            entity_type=entity_type,
            similarity_threshold=similarity_threshold,
        )
        return [
            RetrievedEntity(
                id=entity.id,
                entity_type=entity.entity_type,
                name=entity.name,
                description=entity.description,
                meta=entity.meta,
                similarity_score=similarity_score,
            )
            for entity, similarity_score in entities
        ]

    def search_similar_entities(
        self,
        query: QueryBundle,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        entity_type: EntityType = EntityType.original,
        similarity_threshold: Optional[float] = None,
        # TODO: Metadata filter
        # TODO: include_metadata, include_metadata_keys, include_embeddings parameters
        db_session: Session = None,
    ) -> List[Tuple[E, float]]:
        if query.query_embedding is None:
            query.query_embedding = self._embedding_model.get_query_embedding(
                query.query_str
            )

        entity_model = self._entity_db_model
        with self._session_scope(db_session) as session:
            if entity_type == EntityType.synopsis:
                return self._search_similar_synopsis_entities(
                    session,
                    entity_model,
                    query.query_embedding,
                    top_k,
                    similarity_threshold,
                )
            else:
                return self._search_similar_original_entities(
                    session,
                    entity_model,
                    query.query_embedding,
                    top_k,
                    nprobe,
                    similarity_threshold,
                )

    def _search_similar_original_entities(
        self,
        db_session: Session,
        entity_model: E,
        query_embedding: List[float],
        top_k: int,
        nprobe: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[E, float]]:
        """
        For original entities, it leverages TiFlash's ANN search to efficiently retrieve the most similar entities
        from a large-scale vector space.

        To optimize retrieval performance on ANN Index, there employ a two-phase retrieval strategy:
        1. Fetch more (nprobe) results from the ANN Index as candidates.
        2. Sort the candidates by distance and get the top-k results.
        """
        nprobe = nprobe or top_k * 10
        subquery = (
            select(
                entity_model.id,
                entity_model.description_vec.cosine_distance(query_embedding).label(
                    "distance"
                ),
            )
            .order_by(asc("distance"))
            .limit(nprobe)
            .subquery("candidates")
        )
        query = (
            select(entity_model, (1 - subquery.c.distance).label("similarity_score"))
            .where(entity_model.id == subquery.c.id)
            .where(entity_model.entity_type == EntityType.original)
        )

        if similarity_threshold is not None:
            distance_threshold = 1 - similarity_threshold
            query = query.where(subquery.c.distance <= distance_threshold)

        query = query.order_by(desc("similarity_score")).limit(top_k)
        return db_session.exec(query).all()

    def _search_similar_synopsis_entities(
        self,
        db_session: Session,
        entity_model: E,
        query_embedding: List[float],
        top_k: int,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[E, float]]:
        """
        For synopsis entities, it leverages TiKV to fetch the synopsis entity quickly by filtering by entity_type,
        because the number of synopsis entities is very small, it is commonly faster than using TiFlash to perform
        ANN search.
        """
        hint = text(f"/*+ read_from_storage(tikv[{entity_model.__tablename__}]) */")
        subquery = (
            select(
                entity_model,
                entity_model.description_vec.cosine_distance(query_embedding).label(
                    "distance"
                ),
            )
            .prefix_with(hint)
            .where(entity_model.entity_type == EntityType.synopsis)
            .order_by(asc("distance"))
            .limit(top_k)
            .subquery("candidates")
        )
        query = select(
            entity_model, (1 - subquery.c.distance).label("similarity_score")
        )

        if similarity_threshold is not None:
            distance_threshold = 1 - similarity_threshold
            query = query.where(subquery.c.distance <= distance_threshold)

        query = query.order_by(desc("similarity_score")).limit(top_k)
        return db_session.exec(query).all()

    def _get_the_most_similar_entity(
        self,
        create: EntityCreate,
        similarity_threshold: float = 0,
        db_session: Optional[Session] = None,
    ) -> Optional[E]:
        query = f"{create.name}: {create.description}"
        similar_entities = self.search_similar_entities(
            QueryBundle(query_str=query), top_k=1, nprobe=10, db_session=db_session
        )

        if len(similar_entities) == 0:
            return None

        most_similar_entity, similarity_score = similar_entities[0]

        # For entity with same name and description.
        if (
            most_similar_entity.name == create.name
            and most_similar_entity.description == create.description
            and len(DeepDiff(most_similar_entity.meta, create.meta)) == 0
        ):
            return most_similar_entity

        # For the most similar entity.
        if similarity_score < similarity_threshold:
            return most_similar_entity

        return None

    # Relationship Basic Operations

    def get_relationship_by_id(
        self, relationship_id: int, db_session: Session = None
    ) -> R:
        stmt = select(self._relationship_db_model).where(
            self._relationship_db_model.id == relationship_id
        )
        with self._session_scope(db_session) as session:
            return session.exec(stmt).first()

    def list_relationships(
        self, filters: RelationshipFilters, db_session: Session = None
    ) -> Sequence[R]:
        stmt = self._build_relationships_query(filters)

        with self._session_scope(db_session) as session:
            return session.exec(stmt).all()

    def _build_relationships_query(self, filters: RelationshipFilters):
        stmt = select(self._relationship_db_model)
        if filters.target_entity_id:
            stmt = stmt.where(
                self._relationship_db_model.target_entity_id == filters.target_entity_id
            )
        if filters.target_entity_id:
            stmt = stmt.where(
                self._relationship_db_model.source_target_id == filters.source_target_id
            )
        if filters.relationship_ids:
            stmt = stmt.where(
                self._relationship_db_model.id.in_(filters.relationship_ids)
            )
        if filters.search:
            stmt = stmt.where(
                or_(
                    self._relationship_db_model.name.like(f"%{filters.search}%"),
                    self._relationship_db_model.description.like(f"%{filters.search}%"),
                )
            )
        if filters.chunk_ids:
            stmt = stmt.where(
                self._relationship_db_model.chunk_id.in_(filters.chunk_ids)
            )
        return stmt

    def create_relationship(
        self,
        source_entity: E,
        target_entity: E,
        description: Optional[str] = None,
        metadata: Optional[dict] = {},
        commit: bool = True,
        db_session: Session = None,
    ) -> R:
        """
        Create a relationship between two entities.
        """
        description_vec = self._get_relationship_description_embedding(
            source_entity.name,
            source_entity.description,
            target_entity.name,
            target_entity.description,
            description,
        )
        relationship = self._relationship_db_model(
            source_entity=source_entity,
            target_entity=target_entity,
            description=description,
            description_vec=description_vec,
            meta=metadata,
            chunk_id=metadata["chunk_id"] if "chunk_id" in metadata else None,
            document_id=metadata["document_id"] if "document_id" in metadata else None,
        )

        with self._session_scope(db_session, commit) as session:
            session.add(relationship)
            return relationship

    def update_relationship(
        self,
        relationship: R,
        update: RelationshipUpdate,
        commit: bool = True,
        db_session: Session = None,
    ) -> R:
        for key, value in update.items():
            if value is None:
                continue
            setattr(relationship, key, value)
            flag_modified(relationship, key)

        # Update embeddings.
        relationship.description_vec = self._get_relationship_description_embedding(
            relationship.source_entity.name,
            relationship.source_entity.description,
            relationship.target_entity.name,
            relationship.target_entity.description,
            relationship.description,
        )

        with self._session_scope(db_session, commit) as session:
            session.add(relationship)
            session.refresh(relationship)
        return relationship

    def delete_relationship(
        self, relationship: R, commit: bool = True, db_session: Session = None
    ):
        with self._session_scope(db_session, commit) as session:
            session.delete(relationship)

    def clear_orphan_entities(self):
        raise NotImplementedError()

    # Relationship Retrieve Operations

    def retrieve_relationships(
        self,
        query: str,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        similarity_threshold: Optional[float] = 0,
    ) -> List[RetrievedRelationship]:
        relationships = self.search_similar_relationships(
            query=QueryBundle(query_str=query),
            top_k=top_k,
            nprobe=nprobe,
            similarity_threshold=similarity_threshold,
        )
        return [
            RetrievedRelationship(
                id=relationship.id,
                source_entity_id=relationship.source_entity_id,
                target_entity_id=relationship.target_entity_id,
                description=relationship.description,
                rag_description=f"{relationship.source_entity.name} -> {relationship.description} -> {relationship.target_entity.name}",
                meta=relationship.meta,
                weight=relationship.weight,
                last_modified_at=relationship.last_modified_at,
                similarity_score=similarity_score,
            )
            for relationship, similarity_score in relationships
        ]

    def search_similar_relationships(
        self,
        query: QueryBundle,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        distance_range: Optional[Tuple[float, float]] = None,
        weight_threshold: Optional[float] = None,
        exclude_relationship_ids: Optional[List[str]] = None,
        source_entity_ids: Optional[List[int]] = None,
        target_entity_ids: Optional[List[int]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        db_session: Session = None,
    ) -> List[Tuple[R, float]]:
        if query.query_embedding is None:
            query.query_embedding = self._embedding_model.get_query_embedding(
                query.query_str
            )

        nprobe = nprobe or top_k * 10

        subquery = (
            select(
                self._relationship_db_model.id.label("relationship_id"),
                self._relationship_db_model.description_vec.cosine_distance(
                    query.query_embedding
                ).label("embedding_distance"),
            )
            .order_by(asc("embedding_distance"))
            .limit(nprobe)
            .subquery()
        )
        query = (
            select(
                self._relationship_db_model,
                (1 - subquery.c.embedding_distance).label("similarity_score"),
            )
            .join(
                subquery, self._relationship_db_model.id == subquery.c.relationship_id
            )
            .order_by(desc("similarity_score"))
            .limit(top_k)
        )

        if similarity_threshold is not None:
            distance_threshold = 1 - similarity_threshold
            query = query.where(subquery.c.embedding_distance <= distance_threshold)

        if distance_range is not None:
            query = query.where(
                text(
                    "embedding_distance >= :min_distance AND embedding_distance <= :max_distance"
                )
            ).params(min_distance=distance_range[0], max_distance=distance_range[1])

        if weight_threshold is not None:
            query = query.where(subquery.c.similarity_score >= weight_threshold)

        if exclude_relationship_ids is not None and len(exclude_relationship_ids) > 0:
            query = query.where(
                self._relationship_db_model.id.notin_(exclude_relationship_ids)
            )

        if source_entity_ids is not None and len(source_entity_ids) > 0:
            query = query.where(
                self._relationship_db_model.source_entity_id.in_(source_entity_ids)
            )

        if target_entity_ids is not None and len(target_entity_ids) > 0:
            query = query.where(
                self._relationship_db_model.target_entity_id.in_(target_entity_ids)
            )

        if metadata_filters is not None and len(metadata_filters) > 0:
            for key, value in metadata_filters.items():
                json_path = f"$.{key}"
                if isinstance(value, (list, tuple, set)):
                    value_json = json.dumps(list(value))
                    query = query.where(
                        text(f"JSON_CONTAINS(meta->'$.{key}', :value)")
                    ).params(value=value_json)
                else:
                    query = query.where(
                        text("JSON_EXTRACT(meta, :path) = :value")
                    ).params(path=json_path, value=json.dumps(value))

        with self._session_scope(db_session) as session:
            rows = session.exec(query).all()
            return [(row[0], row.similarity_score) for row in rows]

    # Graph Basic Operations

    def list_relationships_by_ids(
        self, relationship_ids: list[int], db_session: Session = None
    ) -> List[R]:
        stmt = (
            select(self._relationship_db_model)
            .options(
                defer(self._relationship_db_model.description_vec),
                joinedload(self._relationship_db_model.source_entity)
                .defer(self._entity_db_model.description_vec)
                .defer(self._entity_db_model.meta_vec),
                joinedload(self._relationship_db_model.target_entity)
                .defer(self._entity_db_model.description_vec)
                .defer(self._entity_db_model.meta_vec),
            )
            .where(self._relationship_db_model.id.in_(relationship_ids))
        )
        with self._session_scope(db_session) as session:
            return session.exec(stmt).all()
