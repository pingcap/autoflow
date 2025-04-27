import dspy
import logging
import numpy as np
import tidb_vector
from deepdiff import DeepDiff
from typing import List, Optional, Tuple, Dict, Set, Type, Any
from collections import defaultdict

from dspy import Predict
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model

from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
import sqlalchemy
from sqlmodel import Session, asc, func, select, text, SQLModel
from sqlalchemy.orm import aliased, defer, joinedload
from tidb_vector.sqlalchemy import VectorAdaptor
from sqlalchemy import or_, desc

from app.core.db import engine
from app.rag.storage.graph_stores.helpers import (
    get_entity_description_embedding,
    get_relationship_description_embedding,
    calculate_relationship_score,
    get_entity_metadata_embedding,
    get_query_embedding,
    DEFAULT_RANGE_SEARCH_CONFIG,
    DEFAULT_WEIGHT_COEFFICIENT_CONFIG,
    DEFAULT_DEGREE_COEFFICIENT,
)
from app.rag.knowledge.graph.extractor.schema import (
    Entity,
    Relationship,
    SynopsisEntity,
)
from app.models import EntityType

logger = logging.getLogger(__name__)


def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class MergeEntities(dspy.Signature):
    """As a knowledge expert assistant specialized in database technologies, evaluate the two provided entities. These entities have been pre-analyzed and have same name but different descriptions and metadata.
    Please carefully review the detailed descriptions and metadata for both entities to determine if they genuinely represent the same concept or object(entity).
    If you conclude that the entities are identical, merge the descriptions and metadata fields of the two entities into a single consolidated entity.
    If the entities are distinct despite their same name that may be due to different contexts or perspectives, do not merge the entities and return none as the merged entity.

    Considerations: Ensure your decision is based on a comprehensive analysis of the content and context provided within the entity descriptions and metadata.
    Please only response in JSON Format.
    """

    entities: List[Entity] = dspy.InputField(
        desc="List of entities identified from previous analysis."
    )
    merged_entity: Optional[Entity] = dspy.OutputField(
        desc="Merged entity with consolidated descriptions and metadata."
    )


class MergeEntitiesProgram(dspy.Module):
    def __init__(self):
        self.prog = Predict(MergeEntities)

    def forward(self, entities: List[Entity]):
        if len(entities) != 2:
            raise ValueError("The input should contain exactly two entities")

        pred = self.prog(entities=entities)
        return pred


class TiDBGraphStore:
    def __init__(
        self,
        knowledge_base_id: str,
        dspy_lm: dspy.LM,
        session: Optional[Session] = None,
        embed_model: Optional[EmbedType] = None,
        description_similarity_threshold=0.9,
        entity_db_model: Optional[Type[SQLModel]] = None,
        relationship_db_model: Optional[Type[SQLModel]] = None,
        chunk_db_model: Optional[Type[SQLModel]] = None,
    ):
        self.knowledge_base_id = knowledge_base_id
        self._session = session
        self._owns_session = session is None
        if self._session is None:
            self._session = Session(engine)
        self._dspy_lm = dspy_lm

        if embed_model:
            self._embed_model = resolve_embed_model(embed_model)
        else:
            self._embed_model = OpenAIEmbedding(
                model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL
            )

        self.merge_entities_prog = MergeEntitiesProgram()
        self.description_cosine_distance_threshold = (
            1 - description_similarity_threshold
        )
        self._entity_db_model = entity_db_model
        self._relationship_db_model = relationship_db_model
        self._chunk_db_model = chunk_db_model

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
                self._entity_db_model.description_vec,
                tidb_vector.DistanceMetric.COSINE,
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
            self._relationship_model.metadata.create_all(
                engine, tables=[self._relationship_model.__table__]
            )

            # Add HNSW index to accelerate ann queries.
            VectorAdaptor(engine).create_vector_index(
                self._relationship_model.description_vec,
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

    def close_session(self) -> None:
        # Always call this method is necessary to make sure the session is closed
        if self._owns_session:
            self._session.close()

    def save(self, chunk_id, entities_df, relationships_df):
        if entities_df.empty or relationships_df.empty:
            logger.info(
                "Entities or relationships are empty, skip saving to the database"
            )
            return

        if (
            self._session.exec(
                select(self._relationship_model).where(
                    self._relationship_model.meta["chunk_id"] == chunk_id
                )
            ).first()
            is not None
        ):
            logger.info(f"{chunk_id} already exists in the relationship table, skip.")
            return

        entities_name_map = defaultdict(list)
        for _, row in entities_df.iterrows():
            entities_name_map[row["name"]].append(
                self.get_or_create_entity(
                    Entity(
                        name=row["name"],
                        description=row["description"],
                        metadata=row["meta"],
                    ),
                    commit=False,
                )
            )

        def _find_or_create_entity_for_relation(
            name: str, description: str
        ) -> SQLModel:
            _embedding = get_entity_description_embedding(
                name, description, self._embed_model
            )
            # Check entities_name_map first, if not found, then check the database
            for e in entities_name_map.get(name, []):
                if (
                    cosine_distance(e.description_vec, _embedding)
                    < self.description_cosine_distance_threshold
                ):
                    return e
            return self.get_or_create_entity(
                Entity(
                    name=name,
                    description=description,
                    metadata={"status": "need-revised"},
                ),
                commit=False,
            )

        try:
            for _, row in relationships_df.iterrows():
                logger.info(
                    "save entities for relationship %s -> %s -> %s",
                    row["source_entity"],
                    row["relationship_desc"],
                    row["target_entity"],
                )
                source_entity = _find_or_create_entity_for_relation(
                    row["source_entity"], row["source_entity_description"]
                )
                target_entity = _find_or_create_entity_for_relation(
                    row["target_entity"], row["target_entity_description"]
                )

                self.create_relationship(
                    source_entity,
                    target_entity,
                    Relationship(
                        source_entity=source_entity.name,
                        target_entity=target_entity.name,
                        relationship_desc=row["relationship_desc"],
                    ),
                    relationship_metadata=row["meta"],
                    commit=False,
                )

            self._session.commit()
        except Exception as e:
            logger.error(e, exc_info=True)
            self._session.rollback()
            raise e

    def create_relationship(
        self,
        source_entity: SQLModel,
        target_entity: SQLModel,
        relationship: Relationship,
        relationship_metadata: dict = {},
        commit=True,
    ):
        relationship_object = self._relationship_model(
            source_entity=source_entity,
            target_entity=target_entity,
            description=relationship.relationship_desc,
            description_vec=get_relationship_description_embedding(
                source_entity.name,
                source_entity.description,
                target_entity.name,
                target_entity.description,
                relationship.relationship_desc,
                self._embed_model,
            ),
            meta=relationship_metadata,
            document_id=relationship_metadata.get("document_id"),
            chunk_id=relationship_metadata.get("chunk_id"),
        )
        self._session.add(relationship_object)
        if commit:
            self._session.commit()
            self._session.refresh(relationship_object)
        else:
            self._session.flush()

    def get_subgraph_by_relationship_ids(
        self, ids: list[int], **kwargs
    ) -> RetrievedKnowledgeGraph:
        stmt = (
            select(self._relationship_model)
            .where(self._relationship_model.id.in_(ids))
            .options(
                joinedload(self._relationship_model.source_entity),
                joinedload(self._relationship_model.target_entity),
            )
        )
        relationships_set = self._session.exec(stmt)
        entities_set = set()
        relationships = []
        entities = []

        for rel in relationships_set:
            entities_set.add(rel.source_entity)
            entities_set.add(rel.target_entity)
            relationships.append(
                RetrievedRelationship(
                    id=rel.id,
                    knowledge_base_id=self.knowledge_base_id,
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
                    knowledge_base_id=self.knowledge_base_id,
                    name=entity.name,
                    description=entity.description,
                    meta=entity.meta,
                    entity_type=entity.entity_type,
                )
            )

        return RetrievedKnowledgeGraph(
            knowledge_base_id=self.knowledge_base_id,
            entities=entities,
            relationships=relationships,
            **kwargs,
        )

    def get_or_create_entity(self, entity: Entity, commit: bool = True) -> SQLModel:
        # using the cosine distance between the description vectors to determine if the entity already exists
        entity_type = (
            EntityType.synopsis
            if isinstance(entity, SynopsisEntity)
            else EntityType.original
        )
        entity_description_vec = get_entity_description_embedding(
            entity.name,
            entity.description,
            self._embed_model,
        )
        hint = text(
            f"/*+ read_from_storage(tikv[{self._entity_model.__tablename__}]) */"
        )
        result = (
            self._session.query(
                self._entity_model,
                self._entity_model.description_vec.cosine_distance(
                    entity_description_vec
                ).label("distance"),
            )
            .filter(
                self._entity_model.name == entity.name
                and self._entity_model.entity_type == entity_type
            )
            .prefix_with(hint)
            .order_by(asc("distance"))
            .first()
        )
        if (
            result is not None
            and result[1] < self.description_cosine_distance_threshold
        ):
            db_obj = result[0]
            ob_obj_metadata = db_obj.meta
            if (
                db_obj.description == entity.description
                and db_obj.name == entity.name
                and len(DeepDiff(ob_obj_metadata, entity.metadata)) == 0
            ):
                return db_obj
            elif entity_type == EntityType.original:
                # TODO: move to TiDBKnowledgeGraphIndex
                # use LLM to merge the most similar entities
                merged_entity = self._try_merge_entities(
                    [
                        Entity(
                            name=db_obj.name,
                            description=db_obj.description,
                            metadata=ob_obj_metadata,
                        ),
                        Entity(
                            name=entity.name,
                            description=entity.description,
                            metadata=entity.metadata,
                        ),
                    ]
                )
                if merged_entity is not None:
                    db_obj.description = merged_entity.description
                    db_obj.meta = merged_entity.metadata
                    db_obj.description_vec = get_entity_description_embedding(
                        db_obj.name, db_obj.description, self._embed_model
                    )
                    db_obj.meta_vec = get_entity_metadata_embedding(
                        db_obj.meta, self._embed_model
                    )

                    self._session.add(db_obj)
                    if commit:
                        self._session.commit()
                        self._session.refresh(db_obj)
                    else:
                        self._session.flush()
                    return db_obj

        synopsis_info_str = (
            entity.group_info.model_dump()
            if entity_type == EntityType.synopsis
            else None
        )

        db_obj = self._entity_model(
            name=entity.name,
            description=entity.description,
            description_vec=entity_description_vec,
            meta=entity.metadata,
            meta_vec=get_entity_metadata_embedding(entity.metadata, self._embed_model),
            synopsis_info=synopsis_info_str,
            entity_type=entity_type,
        )
        self._session.add(db_obj)
        if commit:
            self._session.commit()
            self._session.refresh(db_obj)
        else:
            self._session.flush()

        return db_obj

    def _try_merge_entities(self, entities: List[Entity]) -> Entity:
        logger.info(f"Trying to merge entities: {entities[0].name}")
        try:
            with dspy.settings.context(lm=self._dspy_lm):
                pred = self.merge_entities_prog(entities=entities)
                return pred.merged_entity
        except Exception as e:
            logger.error(f"Failed to merge entities: {e}", exc_info=True)
            return None

    def retrieve_with_weight(
        self,
        query: str,
        embedding: list,
        depth: int = 2,
        include_meta: bool = False,
        with_degree: bool = False,
        # experimental feature to filter relationships based on meta, can be removed in the future
        relationship_meta_filters: dict = {},
        session: Optional[Session] = None,
    ) -> Tuple[List[RetrievedEntity], List[RetrievedRelationship]]:
        if not embedding:
            assert query, "Either query or embedding must be provided"
            embedding = get_query_embedding(query, self._embed_model)

        relationships, entities = self.search_relationships_weight(
            embedding,
            [],
            [],
            with_degree=with_degree,
            relationship_meta_filters=relationship_meta_filters,
            session=session,
        )

        all_relationships = set(relationships)
        all_entities = set(entities)
        visited_entities = set(e.id for e in entities)
        visited_relationships = set(r.id for r in relationships)

        for _ in range(depth - 1):
            actual_number = 0
            progress = 0
            search_number_each_depth = 10
            for search_config in DEFAULT_RANGE_SEARCH_CONFIG:
                search_ratio = search_config[1]
                search_distance_range = search_config[0]
                remaining_number = search_number_each_depth - actual_number
                # calculate the expected number based search progress
                # It's a accumulative search, so the expected number should be the difference between the expected number and the actual number
                expected_number = (
                    int(
                        (search_ratio + progress) * search_number_each_depth
                        - actual_number
                    )
                    if progress * search_number_each_depth > actual_number
                    else int(search_ratio * search_number_each_depth)
                )
                if expected_number > remaining_number:
                    expected_number = remaining_number
                if remaining_number <= 0:
                    break

                new_relationships, new_entities = self.search_relationships_weight(
                    embedding,
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

        synopsis_entities = self.fetch_similar_entities(
            embedding, top_k=2, entity_type=EntityType.synopsis, session=session
        )
        all_entities.update(synopsis_entities)

        related_doc_ids = set()
        for r in all_relationships:
            if "doc_id" not in r.meta:
                continue
            related_doc_ids.add(r.meta["doc_id"])

        entities = [
            RetrievedEntity(
                id=e.id,
                knowledge_base_id=self.knowledge_base_id,
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
                knowledge_base_id=self.knowledge_base_id,
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

    # Function to fetch degrees for entities
    def fetch_entity_degrees(
        self,
        entity_ids: List[int],
        session: Optional[Session] = None,
    ) -> Dict[int, Dict[str, int]]:
        degrees = {
            entity_id: {"in_degree": 0, "out_degree": 0} for entity_id in entity_ids
        }
        session = session or self._session

        try:
            # Fetch out-degrees
            out_degree_query = (
                session.query(
                    self._relationship_model.source_entity_id,
                    func.count(self._relationship_model.id).label("out_degree"),
                )
                .filter(self._relationship_model.source_entity_id.in_(entity_ids))
                .group_by(self._relationship_model.source_entity_id)
            ).all()

            for row in out_degree_query:
                degrees[row.source_entity_id]["out_degree"] = row.out_degree

            # Fetch in-degrees
            in_degree_query = (
                session.query(
                    self._relationship_model.target_entity_id,
                    func.count(self._relationship_model.id).label("in_degree"),
                )
                .filter(self._relationship_model.target_entity_id.in_(entity_ids))
                .group_by(self._relationship_model.target_entity_id)
            ).all()

            for row in in_degree_query:
                degrees[row.target_entity_id]["in_degree"] = row.in_degree
        except Exception as e:
            logger.error(e)

        return degrees

    def search_relationships_weight(
        self,
        embedding: List[float],
        visited_relationships: Set[int],
        visited_entities: Set[int],
        distance_range: Tuple[float, float] = (0.0, 1.0),
        limit: int = 100,
        weight_coefficient_config: List[
            Tuple[Tuple[int, int], float]
        ] = DEFAULT_WEIGHT_COEFFICIENT_CONFIG,
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
                self._relationship_model,
                self._relationship_model.description_vec.cosine_distance(
                    embedding
                ).label("embedding_distance"),
            )
            .options(defer(self._relationship_model.description_vec))
            .order_by(asc("embedding_distance"))
            .limit(limit * 10)
        ).subquery()

        relationships_alias = aliased(self._relationship_model, subquery)

        query = (
            select(relationships_alias, text("embedding_distance"))
            .options(
                defer(relationships_alias.description_vec),
                joinedload(relationships_alias.source_entity)
                .defer(self._entity_model.meta_vec)
                .defer(self._entity_model.description_vec),
                joinedload(relationships_alias.target_entity)
                .defer(self._entity_model.meta_vec)
                .defer(self._entity_model.description_vec),
            )
            .where(relationships_alias.weight >= 0)
        )

        if relationship_meta_filters:
            for k, v in relationship_meta_filters.items():
                query = query.where(relationships_alias.meta[k] == v)

        if visited_relationships:
            query = query.where(
                self._relationship_model.id.notin_(visited_relationships)
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
                self._relationship_model.source_entity_id.in_(visited_entities)
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
                weight_coefficient_config,
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

    def fetch_similar_entities_by_post_filter(
        self,
        embedding: list,
        top_k: int = 5,
        entity_type: EntityType = EntityType.original,
        session: Optional[Session] = None,
        post_filter_multiplier: int = 10,
    ):
        new_entity_set = set()
        session = session or self._session

        # Create a subquery with a larger limit and include the distance
        subquery = (
            select(
                self._entity_model,
                self._entity_model.description_vec.cosine_distance(embedding).label(
                    "distance"
                ),
            )
            .order_by(asc("distance"))
            .limit(
                post_filter_multiplier * top_k
                if entity_type != EntityType.original
                else top_k
            )
            .subquery()
        )

        # Apply filter only for non-original entity types
        query = (
            select(self._entity_model)
            .where(subquery.c.entity_type == entity_type)
            .order_by(asc(subquery.c.distance))
            .limit(top_k)
        )

        for row in session.exec(query).all():
            new_entity_set.add(row)

        return new_entity_set

    def fetch_similar_entities(
        self,
        embedding: list,
        top_k: int = 10,
        entity_type: EntityType = EntityType.original,
        session: Optional[Session] = None,
    ):
        new_entity_set = set()

        # Retrieve entities based on their ID and similarity to the embedding
        session = session or self._session

        query = select(self._entity_model)

        if entity_type == EntityType.synopsis:
            query = query.where(self._entity_model.entity_type == entity_type)
            hint = text("/*+ read_from_storage(tikv[entities]) */")
            query = query.prefix_with(hint)

        query = query.order_by(
            self._entity_model.description_vec.cosine_distance(embedding)
        ).limit(top_k)

        # Debug: Print the SQL query
        """
        from sqlalchemy.dialects import mysql
        compiled_query = query.compile(
            dialect=mysql.dialect(), compile_kwargs={"literal_binds": True}
        )
        print(f"Debug - SQL Query: {compiled_query}")
        """

        for entity in session.exec(query).all():
            new_entity_set.add(entity)

        return new_entity_set

    def retrieve_graph_data(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve related entities and relationships using semantic search.

        Args:
            query_text: The search query text
            top_k: Maximum number of results to return for each type
            similarity_threshold: Minimum similarity score threshold

        Returns:
            Dictionary containing:
            - entities: List of similar entities with similarity scores
            - relationships: List of similar relationships with similarity scores
        """
        query_embedding = get_query_embedding(query_text, self._embed_model)

        # Query similar entities
        entity_query = (
            select(
                self._entity_model,
                (
                    1
                    - self._entity_model.description_vec.cosine_distance(
                        query_embedding
                    )
                ).label("similarity"),
            )
            .options(
                defer(self._entity_model.description_vec),
                defer(self._entity_model.meta_vec),
            )
            .order_by(desc("similarity"))
            .limit(top_k * 2)  # Fetch more results to account for filtering
        )

        # Query similar relationships
        relationship_query = (
            select(
                self._relationship_model,
                (
                    1
                    - self._relationship_model.description_vec.cosine_distance(
                        query_embedding
                    )
                ).label("similarity"),
            )
            .options(
                defer(self._relationship_model.description_vec),
                joinedload(self._relationship_model.source_entity)
                .defer(self._entity_model.meta_vec)
                .defer(self._entity_model.description_vec),
                joinedload(self._relationship_model.target_entity)
                .defer(self._entity_model.meta_vec)
                .defer(self._entity_model.description_vec),
            )
            .order_by(desc("similarity"))
            .limit(top_k * 2)  # Fetch more results to account for filtering
        )

        # Execute both queries
        entities = []
        relationships = []

        for entity, similarity in self._session.exec(entity_query).all():
            if similarity >= similarity_threshold and len(entities) < top_k:
                entities.append(
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "description": entity.description,
                        "metadata": entity.meta,
                        "similarity_score": similarity,
                    }
                )

        for relationship, similarity in self._session.exec(relationship_query).all():
            if similarity >= similarity_threshold and len(relationships) < top_k:
                relationships.append(
                    {
                        "id": relationship.id,
                        "relationship": relationship.description,
                        "source_entity": {
                            "id": relationship.source_entity.id,
                            "name": relationship.source_entity.name,
                            "description": relationship.source_entity.description,
                        },
                        "target_entity": {
                            "id": relationship.target_entity.id,
                            "name": relationship.target_entity.name,
                            "description": relationship.target_entity.description,
                        },
                        "similarity_score": similarity,
                    }
                )

        return {"entities": entities, "relationships": relationships}

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
            node_ids: List of source node IDs (representing similar entities)
            query: Search query for relevant relationships
            max_depth: Maximum depth for relationship traversal
            max_neighbors: Maximum number of total neighbor paths to return
            similarity_threshold: Minimum similarity score threshold

        Returns:
            Dictionary containing most relevant paths from source nodes to neighbors
        """
        query_embedding = get_query_embedding(query, self._embed_model)
        # Get all source entities
        source_entities = self._session.exec(
            select(self._entity_model)
            .options(
                defer(self._entity_model.description_vec),
                defer(self._entity_model.meta_vec),
            )
            .where(self._entity_model.id.in_(entities_ids))
        ).all()

        # Track visited nodes and discovered paths
        all_visited = set(entities_ids)
        current_level_nodes = set(entities_ids)
        neighbors = []  # Store all discovered paths with their relevance scores

        for depth in range(max_depth):
            if not current_level_nodes:
                break

            # Query relationships for current level
            relationships = self._session.exec(
                select(
                    self._relationship_model,
                    (
                        1
                        - self._relationship_model.description_vec.cosine_distance(
                            query_embedding
                        )
                    ).label("similarity"),
                )
                .options(
                    defer(self._relationship_model.description_vec),
                    joinedload(self._relationship_model.source_entity)
                    .defer(self._entity_model.meta_vec)
                    .defer(self._entity_model.description_vec),
                    joinedload(self._relationship_model.target_entity)
                    .defer(self._entity_model.meta_vec)
                    .defer(self._entity_model.description_vec),
                )
                .where(
                    or_(
                        self._relationship_model.source_entity_id.in_(
                            current_level_nodes
                        ),
                        self._relationship_model.target_entity_id.in_(
                            current_level_nodes
                        ),
                    )
                )
                .order_by(desc("similarity"))
                .limit(max_neighbors * 2)  # Fetch more results to account for filtering
            ).all()

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

    def get_chunks_by_relationships(
        self,
        relationships_ids: List[int],
        session: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """Get chunks for a list of relationships.

        Args:
            relationships: List of relationship objects
            session: Optional database session

        Returns:
            List of dictionaries containing chunk information:
            - text: chunk text content
            - document_id: associated document id
            - meta: chunk metadata
        """
        session = session or self._session

        relationships = session.exec(
            select(self._relationship_model).where(
                self._relationship_model.id.in_(relationships_ids)
            )
        ).all()

        # Extract chunk IDs from relationships
        chunk_ids = {
            rel.meta.get("chunk_id")
            for rel in relationships
            if rel.meta.get("chunk_id") is not None
        }

        if not chunk_ids:
            return []

        # Query chunks
        chunks = session.exec(
            select(self._chunk_model).where(self._chunk_model.id.in_(chunk_ids))
        ).all()

        return [
            {
                "id": chunk.id,
                "text": chunk.text,
                "document_id": chunk.document_id,
                "meta": {
                    "language": chunk.meta.get("language"),
                    "product": chunk.meta.get("product"),
                    "resource": chunk.meta.get("resource"),
                    "source_uri": chunk.meta.get("source_uri"),
                    "tidb_version": chunk.meta.get("tidb_version"),
                },
            }
            for chunk in chunks
        ]

        def get_entity(self, session: Session, entity_id: int) -> Optional[SQLModel]:
        return session.get(self._entity_db_model, entity_id)

    def update_entity(
        self, session: Session, entity: SQLModel, new_entity: dict
    ) -> SQLModel:
        old_entity_dict = entity.screenshot()
        for key, value in new_entity.items():
            if value is not None:
                setattr(entity, key, value)
                flag_modified(entity, key)
        entity.description_vec = get_entity_description_embedding(
            entity.name, entity.description, self._embed_model
        )
        entity.meta_vec = get_entity_metadata_embedding(entity.meta, self._embed_model)
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
                self._embed_model,
            )
            session.add(relationship)
        session.commit()
        session.refresh(entity)
        new_entity_dict = entity.screenshot()
        create_staff_action_log(
            session, "update", "entity", entity.id, old_entity_dict, new_entity_dict
        )
        return entity

    def get_entity_subgraph(
        self, session: Session, entity: SQLModel
    ) -> Tuple[list, list]:
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

    def get_relationship(
        self, session: Session, relationship_id: int
    ) -> Optional[SQLModel]:
        return session.get(self._relationship_db_model, relationship_id)

    def update_relationship(
        self, session: Session, relationship: SQLModel, new_relationship: dict
    ) -> SQLModel:
        old_relationship_dict = relationship.screenshot()
        for key, value in new_relationship.items():
            if value is not None:
                setattr(relationship, key, value)
                flag_modified(relationship, key)
        relationship.description_vec = get_relationship_description_embedding(
            relationship.source_entity.name,
            relationship.source_entity.description,
            relationship.target_entity.name,
            relationship.target_entity.description,
            relationship.description,
            self._embed_model,
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

    def search_similar_entities(
        self, session: Session, query: str, top_k: int = 10
    ) -> list:
        embedding = get_query_embedding(query, self._embed_model)
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
        # with session.begin():
        synopsis_entity = self._entity_db_model(
            name=name,
            description=description,
            description_vec=get_entity_description_embedding(
                name, description, self._embed_model
            ),
            meta=meta,
            meta_vec=get_entity_metadata_embedding(meta, self._embed_model),
            entity_type=EntityType.synopsis,
            synopsis_info={
                "entities": related_entities_ids,
                "topic": topic,
            },
        )
        session.add(synopsis_entity)
        graph_store = TiDBGraphStore(
            knowledge_base=self.knowledge_base_id,
            dspy_lm=None,
            session=session,
            embed_model=self._embed_model,
            entity_db_model=self._entity_db_model,
            relationship_db_model=self._relationship_db_model,
        )
        for related_entity in session.exec(
            select(self._entity_db_model).where(
                self._entity_db_model.id.in_(related_entities_ids)
            )
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
