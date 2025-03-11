import logging
import numpy as np
import tidb_vector
import sqlalchemy
from deepdiff import DeepDiff
from typing import List, Optional, Tuple, Dict, Set, Type, Sequence

from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from sqlalchemy.orm import aliased, defer, joinedload
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, asc, func, select, text, SQLModel, or_
from tidb_vector.sqlalchemy import VectorAdaptor
from sqlalchemy import desc

from app.core.db import engine
from app.models.chunk import get_kb_chunk_model
from app.models.entity import get_kb_entity_model
from app.models.relationship import get_kb_relationship_model
from app.rag.indices.knowledge_graph.graph_store.helpers import (
    get_entity_description_embedding,
    get_relationship_description_embedding,
    calculate_relationship_score,
    get_entity_metadata_embedding,
    get_query_embedding,
    DEFAULT_RANGE_SEARCH_CONFIG,
    DEFAULT_WEIGHT_COEFFICIENTS,
    DEFAULT_DEGREE_COEFFICIENT,
)
from app.rag.indices.knowledge_graph.schema import (
    EntityCreate,
    EntityDegree,
    EntityFilters,
    EntityUpdate,
    RelationshipUpdate,
    RelationshipFilters,
)
from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.retrievers.knowledge_graph.schema import (
    RetrievedEntity,
    RetrievedRelationship,
    RetrievedKnowledgeGraph,
)
from app.models import (
    Entity as DBEntity,
    Relationship as DBRelationship,
    KnowledgeBase,
)
from app.models import EntityType

logger = logging.getLogger(__name__)


def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class TiDBGraphStore:
    def __init__(
        self,
        db_session: Session,
        knowledge_base: KnowledgeBase,
        embed_model: EmbedType,
        entity_model: Type[SQLModel],
        relationship_model: Type[SQLModel],
        chunk_model: Type[SQLModel],
        entity_distance_threshold: Optional[float] = 0.1,
    ):
        self._db_session = db_session
        self._embed_model = resolve_embed_model(embed_model)
        self._entity_db_model = entity_model
        self._relationship_db_model = relationship_model
        self._chunk_db_model = chunk_model
        self._knowledge_base = knowledge_base
        self._entity_distance_threshold = entity_distance_threshold

    @classmethod
    def from_knowledge_base(
        cls, knowledge_base: KnowledgeBase, db_session: Session
    ) -> "TiDBGraphStore":
        embed_model = get_kb_embed_model(db_session, knowledge_base)
        entity_db_model = get_kb_entity_model(knowledge_base)
        relationship_db_model = get_kb_relationship_model(knowledge_base)
        chunk_db_model = get_kb_chunk_model(knowledge_base)
        return cls(
            db_session=db_session,
            knowledge_base=knowledge_base,
            embed_model=embed_model,
            entity_model=entity_db_model,
            relationship_model=relationship_db_model,
            chunk_model=chunk_db_model,
        )

    # Schema Operations

    def ensure_table_schema(self) -> None:
        inspector = sqlalchemy.inspect(engine)
        existed_table_names = inspector.get_table_names()
        entities_table_name = self._entity_db_model.__tablename__
        relationships_table_name = self._relationship_db_model.__tablename__

        if entities_table_name not in existed_table_names:
            self._entity_db_model.metadata.create_all(
                engine, tables=[self._entity_db_model.__table__]
            )

            # Add HNSW index to accelerate ann queries.
            VectorAdaptor(engine).create_vector_index(
                self._entity_db_model.description_vec, tidb_vector.DistanceMetric.COSINE
            )
            VectorAdaptor(engine).create_vector_index(
                self._entity_db_model.meta_vec, tidb_vector.DistanceMetric.COSINE
            )

            logger.info(
                f"Entities table <{entities_table_name}> has been created successfully."
            )
        else:
            logger.info(
                f"Entities table <{entities_table_name}> is already exists, not action to do."
            )

        if relationships_table_name not in existed_table_names:
            self._relationship_db_model.metadata.create_all(
                engine, tables=[self._relationship_db_model.__table__]
            )

            # Add HNSW index to accelerate ann queries.
            VectorAdaptor(engine).create_vector_index(
                self._relationship_db_model.description_vec,
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
        inspector = sqlalchemy.inspect(engine)
        existed_table_names = inspector.get_table_names()
        relationships_table_name = self._relationship_db_model.__tablename__
        entities_table_name = self._entity_db_model.__tablename__

        if relationships_table_name in existed_table_names:
            self._relationship_db_model.metadata.drop_all(
                engine, tables=[self._relationship_db_model.__table__]
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
                engine, tables=[self._entity_db_model.__table__]
            )
            logger.info(
                f"Entities table <{entities_table_name}> has been dropped successfully."
            )
        else:
            logger.info(
                f"Entities table <{entities_table_name}> is not existed, not action to do."
            )

    # Entity Basic Operations

    def fetch_entities_page(
        self,
        filters: Optional[EntityFilters] = EntityFilters(),
        params: Params = Params(),
    ) -> Page[SQLModel]:
        stmt = self._build_entities_query(filters)
        return paginate(self._db_session, stmt, params)

    def list_entities(
        self, filters: Optional[EntityFilters] = EntityFilters()
    ) -> Sequence[SQLModel]:
        stmt = self._build_entities_query(filters)
        return self._db_session.exec(stmt).all()

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

    def get_entity_by_id(self, entity_id: int) -> Type[SQLModel]:
        return self._db_session.get(self._entity_db_model, entity_id)

    def must_get_entity_by_id(self, entity_id: int) -> Type[SQLModel]:
        entity = self.get_entity_by_id(entity_id)
        if entity is None:
            raise ValueError(f"Entity <{entity_id}> does not exist")
        return entity

    def create_entity(self, create: EntityCreate, commit: bool = True) -> SQLModel:
        desc_vec = get_entity_description_embedding(
            create.name, create.description, self._embed_model
        )
        meta_vec = get_entity_metadata_embedding(create.meta, self._embed_model)
        entity = self._entity_db_model(
            name=create.name,
            entity_type=EntityType.original,
            description=create.description,
            description_vec=desc_vec,
            meta=create.meta,
            meta_vec=meta_vec,
        )

        self._db_session.add(entity)
        if commit:
            self._db_session.commit()
            self._db_session.refresh(entity)
        else:
            self._db_session.flush()
        return entity

    def find_or_create_entity(
        self,
        create: EntityCreate,
        commit: bool = True,
    ) -> SQLModel:
        most_similar_entity = self._get_the_most_similar_entity(create)

        if most_similar_entity is not None:
            return most_similar_entity

        return self.create_entity(create, commit=commit)

    def update_entity(
        self, entity: Type[SQLModel], update: EntityUpdate, commit: bool = True
    ) -> Type[SQLModel]:
        for key, value in update.model_dump().items():
            if value is None:
                continue
            setattr(entity, key, value)
            flag_modified(entity, key)

        entity.description_vec = get_entity_description_embedding(
            entity.name, entity.description, self._embed_model
        )
        if update.meta is not None:
            entity.meta_vec = get_entity_metadata_embedding(
                entity.meta, self._embed_model
            )
        self._db_session.add(entity)

        # Update linked relationships.
        linked_relationships = self.list_relationships_by_connected_entity(entity.id)
        for relationship in linked_relationships:
            self.update_relationship(relationship, RelationshipUpdate(), commit)

        if commit:
            self._db_session.commit()
            self._db_session.refresh(entity)
        else:
            self._db_session.flush()
        return entity

    def delete_entity(self, entity: Type[SQLModel], commit: bool = True):
        # Delete linked relationships.
        linked_relationships = self.list_entity_connected_relationships(entity.id)
        for relationship in linked_relationships:
            self._db_session.delete(relationship)

        self._db_session.delete(entity)
        if commit:
            self._db_session.commit()
        else:
            self._db_session.flush()

    def calc_entity_out_degree(self, entity_id: int) -> Optional[int]:
        stmt = select(func.count(self._relationship_db_model.id)).where(
            self._relationship_db_model.source_entity_id == entity_id
        )
        return self._db_session.exec(stmt).one()

    def calc_entity_in_degree(self, entity_id: int) -> Optional[int]:
        stmt = select(func.count(self._relationship_db_model.id)).where(
            self._relationship_db_model.target_entity_id == entity_id
        )
        return self._db_session.exec(stmt).one()

    def calc_entities_degrees(self, entity_ids: List[int]) -> List[EntityDegree]:
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

        results = self._db_session.exec(stmt).all()
        return [
            EntityDegree(
                entity_id=r.id,
                in_degree=r.in_degree,
                out_degree=r.out_degree,
                degrees=r.in_degree + r.out_degree,
            )
            for r in results
        ]

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
            query=query,
            top_k=top_k,
            nprobe=nprobe,
            entity_type=entity_type,
            similarity_threshold=similarity_threshold,
        )
        return [
            RetrievedEntity(
                id=entity.id,
                knowledge_base_id=self._knowledge_base.id,
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
        query: Optional[str] = None,
        query_embedding: List[float] = None,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        entity_type: EntityType = EntityType.original,
        similarity_threshold: Optional[float] = None,
        # TODO: Metadata filter
        # TODO: include_metadata, include_metadata_keys, include_embeddings parameters
    ) -> List[Tuple[SQLModel, float]]:
        if query_embedding is None:
            assert (
                query
            ), "One of the parameters of `query` and `query_embedding` must be provided"
            embedding = get_query_embedding(query, self._embed_model)
        else:
            embedding = query_embedding

        distance_threshold = 1 - similarity_threshold
        entity_model = self._entity_db_model
        nprobe = nprobe or top_k * 10

        if entity_type == EntityType.synopsis:
            return self._search_similar_synopsis_entities(
                entity_model, embedding, top_k, distance_threshold
            )
        else:
            return self._search_similar_original_entities(
                entity_model, embedding, top_k, distance_threshold, nprobe
            )

    def _search_similar_original_entities(
        self,
        entity_model: Type[SQLModel],
        query_embedding: List[float],
        top_k: int,
        distance_threshold: float,
        nprobe: int,
    ) -> List[Tuple[SQLModel, float]]:
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
        return self._db_session.exec(query).all()

    def _search_similar_synopsis_entities(
        self,
        entity_model: Type[SQLModel],
        query_embedding: List[float],
        top_k: int,
        distance_threshold: float,
    ) -> List[Tuple[SQLModel, float]]:
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
        return self._db_session.exec(query).all()

    def _get_the_most_similar_entity(
        self,
        create: EntityCreate,
    ) -> Optional[DBEntity]:
        query = f"{create.name}: {create.description}"
        similar_entities = self.search_similar_entities(query, top_k=1, nprobe=10)

        if len(similar_entities) == 0:
            return None

        most_similar_entity = similar_entities[0]

        # For same entity.
        if (
            most_similar_entity.name == create.name
            and most_similar_entity.description == create.description
            and len(DeepDiff(most_similar_entity.meta, create.meta)) == 0
        ):
            return most_similar_entity

        # For the most similar entity.
        if most_similar_entity.distance < self.entity_distance_threshold:
            return most_similar_entity

        return None

    # Relationship Basic Operations

    def get_relationship_by_id(self, relationship_id: int) -> Type[SQLModel]:
        stmt = select(self._relationship_db_model).where(
            self._relationship_db_model.id == relationship_id
        )
        return self._db_session.exec(stmt).first()

    def fetch_relationships_page(
        self, filters: RelationshipFilters, params: Params
    ) -> Page[Type[SQLModel]]:
        stmt = self._build_relationships_query(filters)
        return paginate(self._db_session, stmt, params)

    def list_relationships(self, filters: RelationshipFilters) -> Sequence[SQLModel]:
        stmt = self._build_relationships_query(filters)
        return self._db_session.exec(stmt).all()

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
        source_entity: Type[SQLModel] | SQLModel,
        target_entity: Type[SQLModel] | SQLModel,
        description: Optional[str] = None,
        metadata: Optional[dict] = {},
        commit: bool = True,
    ) -> SQLModel:
        """
        Create a relationship between two entities.
        """
        description_vec = get_relationship_description_embedding(
            source_entity.name,
            source_entity.description,
            target_entity.name,
            target_entity.description,
            description,
            self._embed_model,
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

        self._db_session.add(relationship)
        if commit:
            self._db_session.commit()
            self._db_session.refresh(relationship)
        else:
            self._db_session.flush()

        return relationship

    def update_relationship(
        self,
        relationship: Type[SQLModel],
        update: RelationshipUpdate,
        commit: bool = True,
    ) -> Type[SQLModel]:
        for key, value in update.items():
            if value is None:
                continue
            setattr(relationship, key, value)
            flag_modified(relationship, key)

        # Update embeddings.
        relationship.description_vec = get_relationship_description_embedding(
            relationship.source_entity.name,
            relationship.source_entity.description,
            relationship.target_entity.name,
            relationship.target_entity.description,
            relationship.description,
            self._embed_model,
        )

        self._db_session.add(relationship)
        if commit:
            self._db_session.commit()
            self._db_session.refresh(relationship)
        else:
            self._db_session.flush()
        return relationship

    def delete_relationship(self, relationship: Type[SQLModel], commit: bool = True):
        self._db_session.delete(relationship)

        if commit:
            self._db_session.commit()
        else:
            self._db_session.flush()

    def clear_orphan_entities(self):
        pass

    # Relationship Chunks Operations

    def exists_chunk_relationships(self, chunk_id: str) -> bool:
        stmt = select(self._relationship_db_model).where(
            self._relationship_db_model.chunk_id == chunk_id
        )
        return self._db_session.exec(stmt).first() is not None

    def batch_get_chunks_by_relationships(
        self, relationships_ids: List[int]
    ) -> List[Type[SQLModel]]:
        """
        Batch get chunks for the provided relationships.
        """
        subquery = (
            select(self._relationship_db_model.chunk_id)
            .where(self._relationship_db_model.id.in_(relationships_ids))
            .subquery()
        )
        stmt = select(self._chunk_db_model).where(self._chunk_db_model.id.in_(subquery))
        return self._db_session.exec(stmt).all()

    # Relationship Retrieve Operations

    def retrieve_relationships(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        similarity_threshold: Optional[float] = 0,
    ) -> List[RetrievedRelationship]:
        relationships = self.search_similar_relationships(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            nprobe=nprobe,
            similarity_threshold=similarity_threshold,
        )
        return [
            RetrievedRelationship(
                id=relationship.id,
                knowledge_base_id=self._knowledge_base.id,
                source_entity_id=relationship.source_entity_id,
                target_entity_id=relationship.target_entity_id,
                description=relationship.description,
                rag_description=f"{relationship.source_entity.name} -> {relationship.description} -> {relationship.target_entity.name}",
                meta=relationship.meta,
                weight=relationship.weight,
                last_modified_at=relationship.last_modified_at,
                similarity_score=relationship.similarity_score,
            )
            for relationship, similarity_score in relationships
        ]

    def search_similar_relationships(
        self,
        query: str,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        query_embedding: List[float] = None,
        similarity_threshold: Optional[float] = 0,
    ) -> List[Tuple[DBRelationship, float]]:
        embedding = query_embedding or get_query_embedding(query, self._embed_model)
        distance_threshold = 1 - similarity_threshold
        nprobe = nprobe or top_k * 10

        subquery = (
            select(
                self._relationship_db_model,
                self._relationship_db_model.description_vec.cosine_distance(
                    embedding
                ).label("distance"),
            )
            .order_by(asc("distance"))
            .limit(nprobe)
            .subquery()
        )
        query = (
            select(subquery, (1 - subquery.c.distance).label("similarity_score"))
            .where(subquery.c.distance <= distance_threshold)
            .order_by(desc("similarity_score"))
            .limit(top_k)
        )

        results = self._db_session.exec(query).all()
        return [(row[0], row.similarity_score) for row in results]

    # Graph Basic Operations

    def list_relationships_by_connected_entity(
        self, entity_id: int
    ) -> List[Type[SQLModel]]:
        stmt = (
            select(self._relationship_db_model)
            .where(
                (self._relationship_db_model.source_entity_id == entity_id)
                | (self._relationship_db_model.target_entity_id == entity_id)
            )
            .options(
                defer(self._relationship_db_model.description_vec),
                joinedload(self._relationship_db_model.source_entity)
                .defer(self._entity_db_model.description_vec)
                .defer(self._entity_db_model.meta_vec),
                joinedload(self._relationship_db_model.target_entity)
                .defer(self._entity_db_model.description_vec)
                .defer(self._entity_db_model.meta_vec),
            )
        )
        return self._db_session.exec(stmt)

    def list_relationships_by_ids(
        self, relationship_ids: list[int], **kwargs
    ) -> List[Type[SQLModel]]:
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
        return self._db_session.exec(stmt).all()

    def _relationships_to_knowledge_graph(
        self, relationships: list[DBRelationship], **kwargs
    ) -> RetrievedKnowledgeGraph:
        entities_set = set()
        relationship_set = set()
        entities = []

        for rel in relationships:
            entities_set.add(rel.source_entity)
            entities_set.add(rel.target_entity)
            relationship_set.add(
                RetrievedRelationship(
                    id=rel.id,
                    knowledge_base_id=self._knowledge_base.id,
                    source_entity_id=rel.source_entity_id,
                    target_entity_id=rel.target_entity_id,
                    description=rel.description,
                    rag_description=f"{rel.source_entity.name} -> {rel.description} -> {rel.target_entity.name}",
                    meta=rel.meta,
                    weight=rel.weight,
                    last_modified_at=rel.last_modified_at,
                )
            )

        for entity in entities_set:
            entities.append(
                RetrievedEntity(
                    id=entity.id,
                    knowledge_base_id=self._knowledge_base.id,
                    name=entity.name,
                    description=entity.description,
                    meta=entity.meta,
                    entity_type=entity.entity_type,
                )
            )

        return RetrievedKnowledgeGraph(
            knowledge_base=self._knowledge_base.to_descriptor(),
            entities=entities,
            relationships=list(relationship_set),
            **kwargs,
        )

    # Knowledge Graph Retrieve Operations

    def traval_knowledge_graph(
        self,
    ) -> RetrievedKnowledgeGraph:
        pass

    def retrieve_knowledge_graph(
        self,
        query: Optional[str],
        query_embedding: Optional[list[float]],
        depth: int = 2,
        include_meta: bool = False,
        with_degree: bool = False,
        metadata_filters: Optional[dict] = None,
    ) -> RetrievedKnowledgeGraph:
        pass

    def retrieve_with_weight(
        self,
        query: str,
        query_embedding: list,
        depth: int = 2,
        include_meta: bool = False,
        with_degree: bool = False,
        # experimental feature to filter relationships based on meta, can be removed in the future
        relationship_meta_filters: dict = {},
        session: Optional[Session] = None,
    ) -> Tuple[List[RetrievedEntity], List[RetrievedRelationship]]:
        if not query_embedding:
            assert query, "Either `query` or `query_embedding` must be provided"
            query_embedding = get_query_embedding(query, self._embed_model)

        relationships, entities = self.search_relationships_weight(
            query_embedding,
            set(),
            set(),
            with_degree=with_degree,
            relationship_meta_filters=relationship_meta_filters,
            session=session,
        )

        all_relationships = set(relationships)
        all_entities = set(entities)
        visited_entities = set(e.id for e in entities)
        visited_relationships = set(r.id for r in relationships)

        fetch_synopsis_entities_num = 2

        for _ in range(depth - 1):
            actual_number = 0
            progress = 0
            search_number_each_level = 10
            for search_config in DEFAULT_RANGE_SEARCH_CONFIG:
                search_ratio = search_config[1]
                search_distance_range = search_config[0]
                remaining_number = search_number_each_level - actual_number
                # calculate the expected number based search progress
                # It's a accumulative search, so the expected number should be the difference between the expected number and the actual number
                expected_number = (
                    int(
                        (search_ratio + progress) * search_number_each_level
                        - actual_number
                    )
                    if progress * search_number_each_level > actual_number
                    else int(search_ratio * search_number_each_level)
                )
                if expected_number > remaining_number:
                    expected_number = remaining_number
                if remaining_number <= 0:
                    break

                new_relationships, new_entities = self.search_relationships_weight(
                    query_embedding,
                    visited_relationships,
                    visited_entities,
                    search_distance_range,
                    rank_n=expected_number,
                    with_degree=with_degree,
                    relationship_meta_filters=relationship_meta_filters,
                    session=session,
                )

                all_relationships.update(new_relationships)
                all_entities.update(new_entities)

                visited_entities.update(e.id for e in new_entities)
                visited_relationships.update(r.id for r in new_relationships)
                actual_number += len(new_relationships)
                # search_ratio == 1 won't count the progress
                if search_ratio != 1:
                    progress += search_ratio

        # Fetch related synopsis entities.
        synopsis_entities = self.search_similar_entities(
            entity_type=EntityType.synopsis,
            query_embedding=query_embedding,
            top_k=fetch_synopsis_entities_num,
        )
        all_entities.update(synopsis_entities)

        entities = [
            RetrievedEntity(
                id=e.id,
                knowledge_base_id=self._knowledge_base.id,
                name=e.name,
                description=e.description,
                meta=e.meta if include_meta else None,
                entity_type=e.entity_type,
            )
            for e in all_entities
        ]
        relationships = [
            RetrievedRelationship(
                id=r.id,
                knowledge_base_id=self._knowledge_base.id,
                source_entity_id=r.source_entity_id,
                target_entity_id=r.target_entity_id,
                rag_description=f"{r.source_entity.name} -> {r.description} -> {r.target_entity.name}",
                description=r.description,
                meta=r.meta,
                weight=r.weight,
                last_modified_at=r.last_modified_at,
            )
            for r in all_relationships
        ]

        return entities, relationships

    def search_relationships_weight(
        self,
        embedding: List[float],
        visited_relationships: Set[int],
        visited_entities: Set[int],
        distance_range: Tuple[float, float] = (0.0, 1.0),
        limit: int = 100,
        weight_coefficients: List[
            Tuple[Tuple[int, int], float]
        ] = DEFAULT_WEIGHT_COEFFICIENTS,
        alpha: float = 1,
        rank_n: int = 10,
        degree_coefficient: float = DEFAULT_DEGREE_COEFFICIENT,
        with_degree: bool = False,
        relationship_meta_filters: Dict = {},
        session: Optional[Session] = None,
    ) -> Tuple[List[SQLModel], List[SQLModel]]:
        # select the relationships to rank
        subquery = (
            select(
                self._relationship_db_model,
                self._relationship_db_model.description_vec.cosine_distance(
                    embedding
                ).label("embedding_distance"),
            )
            .options(defer(self._relationship_db_model.description_vec))
            .order_by(asc("embedding_distance"))
            .limit(limit * 10)
        ).subquery()

        relationships_alias = aliased(self._relationship_db_model, subquery)

        query = (
            select(relationships_alias, text("embedding_distance"))
            .options(
                defer(relationships_alias.description_vec),
                joinedload(relationships_alias.source_entity)
                .defer(self._entity_db_model.meta_vec)
                .defer(self._entity_db_model.description_vec),
                joinedload(relationships_alias.target_entity)
                .defer(self._entity_db_model.meta_vec)
                .defer(self._entity_db_model.description_vec),
            )
            .where(relationships_alias.weight >= 0)
        )

        if relationship_meta_filters:
            for k, v in relationship_meta_filters.items():
                query = query.where(relationships_alias.meta[k] == v)

        if visited_relationships:
            query = query.where(
                self._relationship_db_model.id.notin_(visited_relationships)
            )

        if distance_range != (0.0, 1.0):
            # embedding_distance between the range
            query = query.where(
                text(
                    "embedding_distance >= :min_distance AND embedding_distance <= :max_distance"
                )
            ).params(min_distance=distance_range[0], max_distance=distance_range[1])

        if visited_entities:
            query = query.where(
                self._relationship_db_model.source_entity_id.in_(visited_entities)
            )

        query = query.order_by(asc("embedding_distance")).limit(limit)

        # Order by embedding distance and apply limit
        session = session or self._session
        relationships = session.exec(query).all()

        if len(relationships) <= rank_n:
            relationship_set = set([rel for rel, _ in relationships])
            entity_set = set()
            for r in relationship_set:
                entity_set.add(r.source_entity)
                entity_set.add(r.target_entity)
            return relationship_set, entity_set

        # Fetch degrees if with_degree is True
        if with_degree:
            entity_ids = set()
            for rel, _ in relationships:
                entity_ids.add(rel.source_entity_id)
                entity_ids.add(rel.target_entity_id)
            degrees = self.fetch_entity_degrees(list(entity_ids), session=session)
        else:
            degrees = {}

        # calculate the relationship score based on distance and weight
        ranked_relationships = []
        for relationship, embedding_distance in relationships:
            source_in_degree = (
                degrees[relationship.source_entity_id]["in_degree"]
                if with_degree
                else 0
            )
            target_out_degree = (
                degrees[relationship.target_entity_id]["out_degree"]
                if with_degree
                else 0
            )
            final_score = calculate_relationship_score(
                embedding_distance,
                relationship.weight,
                source_in_degree,
                target_out_degree,
                alpha,
                weight_coefficients,
                degree_coefficient,
                with_degree,
            )
            ranked_relationships.append((relationship, final_score))

        # rank relationships based on the calculated score
        ranked_relationships.sort(key=lambda x: x[1], reverse=True)
        relationship_set = set([rel for rel, score in ranked_relationships[:rank_n]])
        entity_set = set()
        for r in relationship_set:
            entity_set.add(r.source_entity)
            entity_set.add(r.target_entity)

        return list(relationship_set), list(entity_set)

    def retrieve_subgraph_by_similar(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> RetrievedKnowledgeGraph:
        """Retrieve related entities and relationships using semantic search.

        Args:
            query_text: The search query text
            top_k: Maximum number of results to return for each type
            similarity_threshold: Minimum similarity score threshold

        Returns:
            RetrievedKnowledgeGraph containing similar entities and relationships
        """
        query_embedding = get_query_embedding(query_text, self._embed_model)

        # Get similar entities
        entities = self.search_similar_entities(
            query=query_text,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
        )

        # Get similar relationships
        relationships = self.search_similar_relationships(
            query=query_text,
            query_embedding=query_embedding,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
        )

        return RetrievedKnowledgeGraph(
            knowledge_base=self._knowledge_base.to_descriptor(),
            entities=entities,
            relationships=relationships,
        )

    def retrieve_neighbors(
        self,
        entities_ids: List[int],
        query: str,
        max_depth: int = 1,
        max_neighbors: int = 20,
        similarity_threshold: float = 0.7,
    ) -> Dict[str, List[Dict]]:
        """Retrieve most relevant neighbor paths for a group of similar nodes.

        Args:
            entities_ids: List of source node IDs (representing similar entities)
            query: Search query for relevant relationships
            max_depth: Maximum depth for relationship traversal
            max_neighbors: Maximum number of total neighbor paths to return
            similarity_threshold: Minimum similarity score threshold

        Returns:
            Dictionary containing most relevant paths from source nodes to neighbors
        """
        query_embedding = get_query_embedding(query, self._embed_model)

        # Track visited nodes and discovered paths
        all_visited = set(entities_ids)
        current_level_nodes = set(entities_ids)
        neighbors = []  # Store all discovered paths with their relevance scores

        for depth in range(max_depth):
            if not current_level_nodes:
                break

            # Query relationships for current level
            relationships = self.search_similar_relationships(
                query=query,
                query_embedding=query_embedding,
                nprobe=100,
                similarity_threshold=similarity_threshold,
            )

            next_level_nodes = set()

            for rel, similarity in relationships:
                # Skip if similarity is below threshold
                if similarity < similarity_threshold:
                    continue

                # Determine direction and connected entity
                if rel.source_entity_id in current_level_nodes:
                    connected_id = rel.target_entity_id
                else:
                    connected_id = rel.source_entity_id

                # Skip if already visited
                if connected_id in all_visited:
                    continue

                neighbors.append(
                    {
                        "id": rel.id,
                        "relationship": rel.description,
                        "source_entity": {
                            "id": rel.source_entity.id,
                            "name": rel.source_entity.name,
                            "description": rel.source_entity.description,
                        },
                        "target_entity": {
                            "id": rel.target_entity.id,
                            "name": rel.target_entity.name,
                            "description": rel.target_entity.description,
                        },
                        "similarity_score": similarity,
                    }
                )
                next_level_nodes.add(connected_id)
                all_visited.add(connected_id)

            current_level_nodes = next_level_nodes

        # Sort all paths by similarity score and return top max_neighbors
        neighbors.sort(key=lambda x: x["similarity_score"], reverse=True)

        return {"relationships": neighbors[:max_neighbors]}
