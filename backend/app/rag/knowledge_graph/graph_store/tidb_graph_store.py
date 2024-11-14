import dspy
import logging
import numpy as np
from dspy.functional import TypedPredictor
from deepdiff import DeepDiff
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict

from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from sqlmodel import Session, asc, func, select, text
from sqlalchemy.orm import aliased, defer, joinedload
from app.core.db import engine
from app.rag.knowledge_graph.base import KnowledgeGraphStore
from app.rag.knowledge_graph.schema import Entity, Relationship, SynopsisEntity
from app.models import (
    Chunk as DBChunk,
    Entity as DBEntity,
    Relationship as DBRelationship,
    EntityType,
)
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
                select(DBRelationship).where(
                    DBRelationship.meta["chunk_id"] == chunk_id
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
        ) -> DBEntity:
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
                relationship_meatadata=row["meta"],
                commit=False,
            )
        self._session.commit()

    def create_relationship(
        self,
        source_entity: DBEntity,
        target_entity: DBEntity,
        relationship: Relationship,
        relationship_meatadata: dict = {},
        commit=True,
    ) -> DBRelationship:
        relationshipObject = DBRelationship(
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
            meta=relationship_meatadata,
            document_id=relationship_meatadata.get("document_id"),
            chunk_id=relationship_meatadata.get("chunk_id"),
        )
        self._session.add(relationshipObject)
        if commit:
            self._session.commit()

    def get_or_create_entity(self, entity: Entity) -> DBEntity:
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
                DBEntity,
                DBEntity.description_vec.cosine_distance(entity_description_vec).label(
                    "distance"
                ),
            )
            .filter(
                DBEntity.name == entity.name and DBEntity.entity_type == entity_type
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

        db_obj = DBEntity(
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
                # "meta": r.meta,
                "weight": r.weight,
                # "last_modified_at": r.last_modified_at,
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
                    select(DBChunk.text, DBChunk.document_id, DBChunk.meta).where(
                        DBChunk.id.in_(related_doc_ids)
                    )
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
                    DBRelationship.source_entity_id,
                    func.count(DBRelationship.id).label("out_degree"),
                )
                .filter(DBRelationship.source_entity_id.in_(entity_ids))
                .group_by(DBRelationship.source_entity_id)
            ).all()

            for row in out_degree_query:
                degrees[row.source_entity_id]["out_degree"] = row.out_degree

            # Fetch in-degrees
            in_degree_query = (
                session.query(
                    DBRelationship.target_entity_id,
                    func.count(DBRelationship.id).label("in_degree"),
                )
                .filter(DBRelationship.target_entity_id.in_(entity_ids))
                .group_by(DBRelationship.target_entity_id)
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
    ) -> List[DBRelationship]:
        # select the relationships to rank
        subquery = (
            select(
                DBRelationship,
                DBRelationship.description_vec.cosine_distance(embedding).label(
                    "embedding_distance"
                ),
            )
            .options(defer(DBRelationship.description_vec))
            .order_by(asc("embedding_distance"))
            .limit(limit * 10)
        ).subquery()

        relationships_alias = aliased(DBRelationship, subquery)

        query = (
            select(relationships_alias, text("embedding_distance"))
            .options(
                defer(relationships_alias.description_vec),
                joinedload(relationships_alias.source_entity)
                .defer(DBEntity.meta_vec)
                .defer(DBEntity.description_vec),
                joinedload(relationships_alias.target_entity)
                .defer(DBEntity.meta_vec)
                .defer(DBEntity.description_vec),
            )
            .where(relationships_alias.weight >= 0)
        )

        if relationship_meta_filters:
            for k, v in relationship_meta_filters.items():
                query = query.where(relationships_alias.meta[k] == v)

        if visited_relationships:
            query = query.where(DBRelationship.id.notin_(visited_relationships))

        if distance_range != (0.0, 1.0):
            # embedding_distance bewteen the range
            query = query.where(
                text(
                    "embedding_distance >= :min_distance AND embedding_distance <= :max_distance"
                )
            ).params(min_distance=distance_range[0], max_distance=distance_range[1])

        if visited_entities:
            query = query.where(DBRelationship.source_entity_id.in_(visited_entities))

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
                DBEntity,
                DBEntity.description_vec.cosine_distance(embedding).label("distance"),
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
            select(DBEntity)
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

        query = select(DBEntity)

        if entity_type == EntityType.synopsis:
            query = query.where(DBEntity.entity_type == entity_type)
            hint = text("/*+ read_from_storage(tikv[entities]) */")
            query = query.prefix_with(hint)

        query = query.order_by(
            DBEntity.description_vec.cosine_distance(embedding)
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
