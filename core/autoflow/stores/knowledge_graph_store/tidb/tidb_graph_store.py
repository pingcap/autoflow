import json
import logging
import numpy as np
import tidb_vector
import sqlalchemy

from typing import List, Optional, Tuple, Dict, Type, Sequence, Any, Mapping, Collection

from deepdiff import DeepDiff
from sqlalchemy.orm import defer, joinedload
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, asc, func, select, text, or_
from tidb_vector.sqlalchemy import VectorAdaptor
from sqlalchemy import desc, Engine
from autoflow.models.embeddings import EmbeddingModel
from autoflow.db_models.entity import EntityType
from autoflow.stores.knowledge_graph_store import KnowledgeGraphStore
from autoflow.stores.knowledge_graph_store.algorithms.base import GraphSearchAlgorithm
from autoflow.stores.knowledge_graph_store.base import (
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
    RetrievedKnowledgeGraph,
)
from autoflow.stores.schema import QueryBundle

logger = logging.getLogger(__name__)


def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class TiDBKnowledgeGraphStore(KnowledgeGraphStore[E, R, C]):
    def __init__(
        self,
        db_engine: Engine,
        embedding_model: EmbeddingModel,
        entity_db_model: Type[E],
        relationship_db_model: Type[R],
    ):
        self._db_engine = db_engine
        self._embedding_model = embedding_model
        self._entity_db_model = entity_db_model
        self._relationship_db_model = relationship_db_model

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

    # Entity Basic Operations

    # def fetch_entities_page(
    #     self,
    #     filters: Optional[EntityFilters] = EntityFilters(),
    #     params: Params = Params(),
    # ) -> Page[E]:
    #     stmt = self._build_entities_query(filters)
    #     with Session(self._db_engine) as db_session:
    #         return paginate(db_session, stmt, params)

    def list_entities(
        self, filters: Optional[EntityFilters] = EntityFilters()
    ) -> Sequence[E]:
        stmt = self._build_entities_query(filters)
        with Session(self._db_engine) as db_session:
            return db_session.exec(stmt).all()

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
        with Session(self._db_engine) as db_session:
            return db_session.get(self._entity_db_model, entity_id)

    def must_get_entity_by_id(self, entity_id: int) -> Type[E]:
        entity = self.get_entity_by_id(entity_id)
        if entity is None:
            raise ValueError(f"Entity <{entity_id}> does not exist")
        return entity

    def create_entity(self, create: EntityCreate, commit: bool = True) -> E:
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

        with Session(self._db_engine) as db_session:
            db_session.add(entity)
            if commit:
                db_session.commit()
                db_session.refresh(entity)
            else:
                db_session.flush()
            return entity

    def find_or_create_entity(self, create: EntityCreate, commit: bool = True) -> E:
        most_similar_entity = self._get_the_most_similar_entity(create)

        if most_similar_entity is not None:
            return most_similar_entity

        return self.create_entity(create, commit=commit)

    def update_entity(
        self, entity: Type[E], update: EntityUpdate, commit: bool = True
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

        with Session(self._db_engine) as db_session:
            db_session.add(entity)
            # Update linked relationships.
            connected_relationships = self.list_entity_relationships(entity.id)
            for relationship in connected_relationships:
                self.update_relationship(relationship, RelationshipUpdate(), commit)

            if commit:
                db_session.commit()
                db_session.refresh(entity)
            else:
                db_session.flush()
            return entity

    def delete_entity(self, entity: Type[E], commit: bool = True) -> None:
        with Session(self._db_engine) as db_session:
            # Delete linked relationships.
            linked_relationships = self.list_entity_relationships(entity.id)
            for relationship in linked_relationships:
                db_session.delete(relationship)

            db_session.delete(entity)
            if commit:
                db_session.commit()
            else:
                db_session.flush()

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
        with Session(self._db_engine) as db_session:
            return db_session.exec(stmt).all()

    def calc_entity_out_degree(self, entity_id: int) -> Optional[int]:
        stmt = select(func.count(self._relationship_db_model.id)).where(
            self._relationship_db_model.source_entity_id == entity_id
        )
        with Session(self._db_engine) as db_session:
            return db_session.exec(stmt).one()

    def calc_entity_in_degree(self, entity_id: int) -> Optional[int]:
        stmt = select(func.count(self._relationship_db_model.id)).where(
            self._relationship_db_model.target_entity_id == entity_id
        )
        with Session(self._db_engine) as db_session:
            return db_session.exec(stmt).one()

    def calc_entity_degree(self, entity_id: int) -> Optional[int]:
        stmt = select(func.count(self._relationship_db_model.id)).where(
            or_(
                self._relationship_db_model.target_entity_id == entity_id,
                self._relationship_db_model.source_entity_id == entity_id,
            )
        )
        with Session(self._db_engine) as db_session:
            return db_session.exec(stmt).one()

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

        with Session(self._db_engine) as db_session:
            results = db_session.exec(stmt).all()
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
    ) -> List[Tuple[E, float]]:
        if query.query_embedding is None:
            query.query_embedding = self._embedding_model.get_query_embedding(
                query.query_str
            )

        distance_threshold = 1 - similarity_threshold
        entity_model = self._entity_db_model
        nprobe = nprobe or top_k * 10

        if entity_type == EntityType.synopsis:
            return self._search_similar_synopsis_entities(
                entity_model, query.query_embedding, top_k, distance_threshold
            )
        else:
            return self._search_similar_original_entities(
                entity_model, query.query_embedding, top_k, distance_threshold, nprobe
            )

    def _search_similar_original_entities(
        self,
        entity_model: E,
        query_embedding: List[float],
        top_k: int,
        distance_threshold: float,
        nprobe: int,
    ) -> List[Tuple[E, float]]:
        """
        For original entities, it leverages TiFlash's ANN search to efficiently retrieve the most similar entities
        from a large-scale vector space.

        To optimize retrieval performance on ANN Index, there employ a two-phase retrieval strategy:
        1. Fetch more (nprobe) results from the ANN Index as candidates.
        2. Sort the candidates by distance and get the top-k results.
        """
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
            .where(subquery.c.distance <= distance_threshold)
            .where(entity_model.id == subquery.c.id)
            .where(entity_model.entity_type == EntityType.original)
            .order_by(desc("similarity_score"))
            .limit(top_k)
        )
        with Session(self._db_engine) as db_session:
            return db_session.exec(query).all()

    def _search_similar_synopsis_entities(
        self,
        entity_model: E,
        query_embedding: List[float],
        top_k: int,
        distance_threshold: float,
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
        query = (
            select(entity_model, (1 - subquery.c.distance).label("similarity_score"))
            .where(subquery.c.distance <= distance_threshold)
            .order_by(desc("similarity_score"))
            .limit(top_k)
        )
        with Session(self._db_engine) as db_session:
            return db_session.exec(query).all()

    def _get_the_most_similar_entity(
        self,
        create: EntityCreate,
        similarity_threshold: float = 0,
    ) -> Optional[E]:
        query = f"{create.name}: {create.description}"
        similar_entities = self.search_similar_entities(query, top_k=1, nprobe=10)

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

    def get_relationship_by_id(self, relationship_id: int) -> R:
        stmt = select(self._relationship_db_model).where(
            self._relationship_db_model.id == relationship_id
        )
        with Session(self._db_engine) as db_session:
            return db_session.exec(stmt).first()

    # def fetch_relationships_page(
    #     self, filters: RelationshipFilters, params: Params
    # ) -> Page[Type[SQLModel]]:
    #     stmt = self._build_relationships_query(filters)
    #     with Session(self._db_engine) as db_session:
    #         return paginate(db_session, stmt, params)

    def list_relationships(self, filters: RelationshipFilters) -> Sequence[R]:
        stmt = self._build_relationships_query(filters)
        with Session(self._db_engine) as db_session:
            return db_session.exec(stmt).all()

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
        return stmt

    def create_relationship(
        self,
        source_entity: E,
        target_entity: E,
        description: Optional[str] = None,
        metadata: Optional[dict] = {},
        commit: bool = True,
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

        with Session(self._db_engine) as db_session:
            db_session.add(relationship)
            if commit:
                db_session.commit()
                db_session.refresh(relationship)
            else:
                db_session.flush()
            return relationship

    def update_relationship(
        self,
        relationship: R,
        update: RelationshipUpdate,
        commit: bool = True,
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

        with Session(self._db_engine) as db_session:
            db_session.add(relationship)
            if commit:
                db_session.commit()
                db_session.refresh(relationship)
            else:
                db_session.flush()
            return relationship

    def delete_relationship(self, relationship: R, commit: bool = True):
        with Session(self._db_engine) as db_session:
            db_session.delete(relationship)
            if commit:
                db_session.commit()
            else:
                db_session.flush()

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

        if exclude_relationship_ids is not None:
            query = query.where(
                self._relationship_db_model.id.notin_(exclude_relationship_ids)
            )

        if source_entity_ids is not None:
            query = query.where(
                self._relationship_db_model.source_entity_id.in_(source_entity_ids)
            )

        if target_entity_ids is not None:
            query = query.where(
                self._relationship_db_model.target_entity_id.in_(target_entity_ids)
            )

        if metadata_filters:
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

        with Session(self._db_engine) as db_session:
            rows = db_session.exec(query).all()
            return [(row[0], row.similarity_score) for row in rows]

    # Graph Basic Operations

    def list_relationships_by_ids(
        self, relationship_ids: list[int], **kwargs
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
        with Session(self._db_engine) as db_session:
            return db_session.exec(stmt).all()

    def search(
        self,
        query: QueryBundle,
        depth: int = 2,
        include_meta: bool = False,
        meta_filters: dict = {},
        search_algorithm: GraphSearchAlgorithm = None,
    ) -> RetrievedKnowledgeGraph:
        """Search the knowledge graph using configurable search algorithm

        Args:
            query: Query bundle containing search text or embedding
            depth: Maximum search depth in the graph
            include_meta: Whether to include metadata in results
            meta_filters: Filters to apply on metadata
            search_algorithm: Algorithm class to use for graph search

        Returns:
            Retrieved subgraph containing matching entities and relationships
        """
        # Ensure query has embedding
        if query.query_embedding is None and hasattr(self, "_embed_model"):
            query.query_embedding = self._embed_model.get_query_embedding(
                query.query_str
            )

        # Initialize and execute search algorithm
        algorithm = search_algorithm()
        relationships, entities = algorithm.search(
            self,
            query=query,
            depth=depth,
            meta_filters=meta_filters,
        )

        # Construct result graph
        return RetrievedKnowledgeGraph(
            relationships=[
                RetrievedRelationship(
                    id=r.id,
                    source_entity_id=r.source_entity_id,
                    target_entity_id=r.target_entity_id,
                    description=r.description,
                    rag_description=f"{r.source_entity.name} -> {r.description} -> {r.target_entity.name}",
                    meta=r.meta if include_meta else None,
                    weight=r.weight,
                    last_modified_at=r.last_modified_at,
                    similarity_score=r.score if hasattr(r, "score") else None,
                )
                for r in relationships
            ],
            entities=[
                RetrievedEntity(
                    id=e.id,
                    entity_type=e.entity_type,
                    name=e.name,
                    description=e.description,
                    meta=e.meta if include_meta else None,
                )
                for e in entities
            ],
        )
