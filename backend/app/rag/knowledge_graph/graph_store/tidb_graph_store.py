import dspy
import logging
import numpy as np
import tidb_vector
from dspy.functional import TypedPredictor
from deepdiff import DeepDiff
from typing import List, Optional, Tuple, Dict, Set, Type, Any
from collections import defaultdict

from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
import sqlalchemy
from sqlmodel import Session, asc, func, select, text, SQLModel
from sqlalchemy.orm import aliased, defer, joinedload
from tidb_vector.sqlalchemy import VectorAdaptor
from sqlalchemy import or_, desc

from app.core.db import engine
from app.rag.knowledge_graph.base import KnowledgeGraphStore
from app.rag.knowledge_graph.schema import Entity, Relationship, SynopsisEntity
from app.models import (
    Entity as DBEntity,
    Relationship as DBRelationship,
    Chunk as DBChunk,
)
from app.models import EntityType
from app.rag.knowledge_graph.graph_store.helpers import (
    calculate_relationship_score,
    DEFAULT_WEIGHT_COEFFICIENT_CONFIG,
    DEFAULT_RANGE_SEARCH_CONFIG,
    DEFAULT_DEGREE_COEFFICIENT,
    get_query_embedding,
    get_entity_description_embedding,
    get_entity_metadata_embedding,
    get_relationship_description_embedding,
)

logger = logging.getLogger(__name__)


def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class MergeEntities(dspy.Signature):
    """As a knowledge expert assistant specialized in database technologies, evaluate the two provided entities. These entities have been pre-analyzed and have same name but different descriptions and metadata.
    Please carefully review the detailed descriptions and metadata for both entities to determine if they genuinely represent the same concept or object(entity).
    If you conclude that the entities are identical, merge the descriptions and metadata fields of the two entities into a single consolidated entity.
    If the entities are distinct despite their same name that may be due to different contexts or perspectives, do not merge the entities and return none as the merged entity.

    Considerations: Ensure your decision is based on a comprehensive analysis of the content and context provided within the entity descriptions and metadata.
    """

    entities: List[Entity] = dspy.InputField(
        desc="List of entities identified from previous analysis."
    )
    merged_entity: Optional[Entity] = dspy.OutputField(
        desc="Merged entity with consolidated descriptions and metadata."
    )


class MergeEntitiesProgram(dspy.Module):
    def __init__(self):
        self.prog = TypedPredictor(MergeEntities)

    def forward(self, entities: List[Entity]):
        if len(entities) != 2:
            raise ValueError("The input should contain exactly two entities")

        pred = self.prog(entities=entities)
        return pred


class TiDBGraphStore(KnowledgeGraphStore):
    def __init__(
        self,
        dspy_lm: dspy.LM,
        session: Optional[Session] = None,
        embed_model: Optional[EmbedType] = None,
        description_similarity_threshold=0.9,
        entity_db_model: Type[SQLModel] = DBEntity,
        relationship_db_model: Type[SQLModel] = DBRelationship,
        chunk_db_model: Type[SQLModel] = DBChunk,
    ):
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
        self._entity_model = entity_db_model
        self._relationship_model = relationship_db_model
        self._chunk_model = chunk_db_model

    def ensure_table_schema(self) -> None:
        inspector = sqlalchemy.inspect(engine)
        existed_table_names = inspector.get_table_names()
        entities_table_name = self._entity_model.__tablename__
        relationships_table_name = self._relationship_model.__tablename__

        if entities_table_name not in existed_table_names:
            self._entity_model.metadata.create_all(
                engine, tables=[self._entity_model.__table__]
            )

            # Add HNSW index to accelerate ann queries.
            VectorAdaptor(engine).create_vector_index(
                self._entity_model.description_vec, tidb_vector.DistanceMetric.COSINE
            )
            VectorAdaptor(engine).create_vector_index(
                self._entity_model.meta_vec, tidb_vector.DistanceMetric.COSINE
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
        relationships_table_name = self._relationship_model.__tablename__
        entities_table_name = self._entity_model.__tablename__

        if relationships_table_name in existed_table_names:
            self._relationship_model.metadata.drop_all(
                engine, tables=[self._relationship_model.__table__]
            )
            logger.info(
                f"Relationships table <{relationships_table_name}> has been dropped successfully."
            )
        else:
            logger.info(
                f"Relationships table <{relationships_table_name}> is not existed, not action to do."
            )

        if entities_table_name in existed_table_names:
            self._entity_model.metadata.drop_all(
                engine, tables=[self._entity_model.__table__]
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
            )

        for _, row in relationships_df.iterrows():
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

    def get_or_create_entity(self, entity: Entity) -> SQLModel:
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
                    self._session.commit()
                    self._session.refresh(db_obj)
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
        self._session.commit()
        self._session.refresh(db_obj)
        return db_obj

    def _try_merge_entities(self, entities: List[Entity]) -> Entity:
        logger.info(f"Trying to merge entities: {entities[0].name}")
        with dspy.settings.context(lm=self._dspy_lm):
            pred = self.merge_entities_prog(entities=entities)
            return pred.merged_entity

    def retrieve_with_weight(
        self,
        query: str,
        embedding: list,
        depth: int = 2,
        include_meta: bool = False,
        with_degree: bool = False,
        with_chunks: bool = True,
        # experimental feature to filter relationships based on meta, can be removed in the future
        relationship_meta_filters: Dict = {},
        session: Optional[Session] = None,
    ) -> Tuple[list, list, list]:
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
                # seach ratio == 1 won't count the progress
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
            {
                "id": e.id,
                "name": e.name,
                "description": e.description,
                "meta": e.meta if include_meta else None,
                "entity_type": e.entity_type,
            }
            for e in all_entities
        ]
        relationships = [
            {
                "id": r.id,
                "source_entity_id": r.source_entity_id,
                "target_entity_id": r.target_entity_id,
                "description": r.description,
                "rag_description": f"{r.source_entity.name} -> {r.description} -> {r.target_entity.name}",
                "meta": r.meta,
                "weight": r.weight,
                "last_modified_at": r.last_modified_at,
            }
            for r in all_relationships
        ]

        chunks = []
        session = session or self._session
        if with_chunks:
            chunks = [
                # TODO: add last_modified_at
                {"text": c[0], "link": c[1], "meta": c[2]}
                for c in session.exec(
                    select(
                        self._chunk_model.text,
                        self._chunk_model.document_id,
                        self._chunk_model.meta,
                    ).where(self._chunk_model.id.in_(related_doc_ids))
                ).all()
            ]

        return entities, relationships, chunks

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
    ) -> List[SQLModel]:
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

        return relationship_set, entity_set

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
        top_k: int = 5,
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
            .having(text("similarity >= :threshold"))
            .params(threshold=similarity_threshold)
            .limit(top_k)
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
            .having(text("similarity >= :threshold"))
            .params(threshold=similarity_threshold)
            .limit(top_k)
        )

        # Execute both queries
        entities = []
        relationships = []

        for entity, similarity in self._session.exec(entity_query).all():
            entities.append({"entity": entity, "similarity_score": similarity})

        for relationship, similarity in self._session.exec(relationship_query).all():
            relationships.append(
                {
                    "relationship": relationship,
                    "source_entity": relationship.source_entity,
                    "target_entity": relationship.target_entity,
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
        all_paths = []  # Store all discovered paths with their relevance scores

        for depth in range(max_depth):
            if not current_level_nodes:
                break

            # Query relationships for current level
            relationships = self._session.exec(
                select(self._relationship_model)
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
            ).all()

            next_level_nodes = set()

            for rel in relationships:
                # Determine direction and connected entity
                if rel.source_entity_id in current_level_nodes:
                    from_entity = rel.source_entity
                    to_entity = rel.target_entity
                    direction = "outgoing"
                    connected_id = rel.target_entity_id
                else:
                    from_entity = rel.target_entity
                    to_entity = rel.source_entity
                    direction = "incoming"
                    connected_id = rel.source_entity_id

                # Skip if already visited
                if connected_id in all_visited:
                    continue

                # Calculate relationship relevance
                similarity = 1 - cosine_distance(query_embedding, rel.description_vec)

                if similarity >= similarity_threshold:
                    # Find the original source node that led to this path
                    for source_entity in source_entities:
                        if source_entity.id == from_entity.id:
                            path = {
                                "source_entity": {
                                    "id": from_entity.id,
                                    "name": from_entity.name,
                                    "description": from_entity.description,
                                },
                                "path": [
                                    {
                                        "from_entity": {
                                            "id": from_entity.id,
                                            "name": from_entity.name,
                                            "description": from_entity.description,
                                        },
                                        "relationship": {
                                            "id": rel.id,
                                            "description": rel.description,
                                        },
                                        "to_entity": {
                                            "id": to_entity.id,
                                            "name": to_entity.name,
                                            "description": to_entity.description,
                                        },
                                        "direction": direction,
                                        "depth": depth + 1,
                                    }
                                ],
                                "similarity_score": similarity,
                            }
                            all_paths.append(path)
                            next_level_nodes.add(connected_id)
                            all_visited.add(connected_id)

                        # For paths longer than depth 1, find existing paths that end at from_entity
                        elif depth > 0:
                            matching_paths = [
                                p
                                for p in all_paths
                                if p["path"][-1]["to_entity"].id == from_entity.id
                            ]
                            for existing_path in matching_paths:
                                # Create new path by extending existing path
                                new_path = {
                                    "source_entity": existing_path["source_entity"],
                                    "path": existing_path["path"]
                                    + [
                                        {
                                            "from_entity": {
                                                "id": from_entity.id,
                                                "name": from_entity.name,
                                                "description": from_entity.description,
                                            },
                                            "relationship": {
                                                "id": rel.id,
                                                "description": rel.description,
                                            },
                                            "to_entity": {
                                                "id": to_entity.id,
                                                "name": to_entity.name,
                                                "description": to_entity.description,
                                            },
                                            "direction": direction,
                                            "depth": depth + 1,
                                        }
                                    ],
                                    # Combine similarities (you might want to adjust this formula)
                                    "similarity_score": (
                                        existing_path["similarity_score"] + similarity
                                    )
                                    / 2,
                                }
                                all_paths.append(new_path)
                                next_level_nodes.add(connected_id)
                                all_visited.add(connected_id)

            current_level_nodes = next_level_nodes

        # Sort all paths by similarity score and return top max_neighbors
        all_paths.sort(key=lambda x: x["similarity_score"], reverse=True)

        return all_paths[:max_neighbors]


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
            select(self._relationship_model)
            .where(self._relationship_model.id.in_(relationships_ids))
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
            {"text": chunk.text, "document_id": chunk.document_id, "meta": chunk.meta}
            for chunk in chunks
        ]
