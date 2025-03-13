from collections import defaultdict
from typing import List, Set, Tuple, Optional

from autoflow.models import EntityType
from autoflow.storage.graph_store import KnowledgeGraphStore
from .base import KnowledgeGraphRetriever, RelationshipWithScore
from ..base import E, R, EntityDegree

from ...schema import QueryBundle


# The configuration for the weight coefficient
# format: ((min_weight, max_weight), coefficient)
DEFAULT_WEIGHT_COEFFICIENTS = [
    ((0, 100), 0.01),
    ((100, 1000), 0.001),
    ((1000, 10000), 0.0001),
    ((10000, float("inf")), 0.00001),
]

# The configuration for the range search
# format: ((min_distance, max_distance), search_ratio)
# The sum of search ratio should be 1 except some case we want to search as many as possible relationships.
# In this case, we set the search ratio to 1, and the other search ratio sum should be 1
DEFAULT_RANGE_SEARCH_CONFIG = [
    ((0.0, 0.25), 1),
    ((0.25, 0.35), 0.7),
    ((0.35, 0.45), 0.2),
    ((0.45, 0.55), 0.1),
]

DEFAULT_DEGREE_COEFFICIENT = 0.001


class WeightedGraphSearchRetriever(KnowledgeGraphRetriever[E, R]):
    def __init__(
        self,
        kg_store: KnowledgeGraphStore,
        with_degree: bool = False,
        alpha: float = 1,
        weight_coefficients: List[Tuple[float, float]] = None,
        search_range_config: List[Tuple[Tuple[float, float], float]] = None,
        degree_coefficient: float = DEFAULT_DEGREE_COEFFICIENT,
        fetch_synopsis_entities_num: int = 2,
        max_neighbors: int = 10,
    ):
        super().__init__(kg_store)
        self.with_degree = with_degree
        self.alpha = alpha
        self.weight_coefficients = weight_coefficients or DEFAULT_WEIGHT_COEFFICIENTS
        self.search_range_config = search_range_config or DEFAULT_RANGE_SEARCH_CONFIG
        self.degree_coefficient = degree_coefficient
        self.fetch_synopsis_entities_num = fetch_synopsis_entities_num
        self.max_neighbors = max_neighbors

    def search(
        self,
        query_embedding: List[float],
        depth: int = 2,
        metadata_filters: Optional[dict] = None,
    ) -> Tuple[List[RelationshipWithScore[R]], List[E]]:
        visited_relationships = set()
        visited_entities = set()

        new_relationships = self._weighted_search_relationships(
            query_embedding=query_embedding,
            visited_relationships=visited_relationships,
            visited_entities=visited_entities,
            metadata_filters=metadata_filters,
        )

        if len(new_relationships) == 0:
            return [], []

        for rel, score in new_relationships:
            visited_relationships.add(
                RelationshipWithScore(relationship=rel, score=score)
            )
            visited_entities.add(rel.source_entity)
            visited_entities.add(rel.target_entity)

        for _ in range(depth - 1):
            actual_number = 0
            progress = 0
            for search_config in DEFAULT_RANGE_SEARCH_CONFIG:
                search_ratio = search_config[1]
                search_distance_range = search_config[0]
                remaining_number = self.max_neighbors - actual_number
                # calculate the expected number based search progress
                # It's an accumulative search, so the expected number should be the difference between the expected number and the actual number
                expected_number = (
                    int((search_ratio + progress) * self.max_neighbors - actual_number)
                    if progress * self.max_neighbors > actual_number
                    else int(search_ratio * self.max_neighbors)
                )
                if expected_number > remaining_number:
                    expected_number = remaining_number
                if remaining_number <= 0:
                    break

                new_relationships = self._weighted_search_relationships(
                    query_embedding=query_embedding,
                    visited_relationships=visited_relationships,
                    visited_entities=visited_entities,
                    search_distance_range=search_distance_range,
                    top_k=expected_number,
                    metadata_filters=metadata_filters,
                )

                for rel, score in new_relationships:
                    visited_relationships.add(
                        RelationshipWithScore(relationship=rel, score=score)
                    )
                    visited_entities.add(rel.source_entity)
                    visited_entities.add(rel.target_entity)

                actual_number += len(new_relationships)
                # search_ratio == 1 won't count the progress
                if search_ratio != 1:
                    progress += search_ratio

        # Fetch related synopsis entities.
        synopsis_entities = self._kg_store.search_similar_entities(
            query=QueryBundle(query_embedding=query_embedding),
            entity_type=EntityType.synopsis,
            top_k=self.fetch_synopsis_entities_num,
        )
        visited_entities.update(synopsis_entities)

        return list(visited_relationships), list(visited_entities)

    def _weighted_search_relationships(
        self,
        query_embedding: List[float],
        visited_relationships: Set[RelationshipWithScore[R]],
        visited_entities: Set[E],
        search_distance_range: Tuple[float, float] = (0, 1),
        top_k: int = 10,
        metadata_filters: Optional[dict] = None,
    ) -> List[Tuple[R, float]]:
        visited_entity_ids = [e.id for e in visited_entities]
        visited_relationship_ids = [r.relationship.id for r in visited_relationships]
        relationships_with_score = self._kg_store.search_similar_relationships(
            query=QueryBundle(query_embedding=query_embedding),
            distance_range=search_distance_range,
            source_entity_ids=visited_entity_ids,
            exclude_relationship_ids=visited_relationship_ids,
            metadata_filters=metadata_filters,
        )
        return self._rerank_relationships(
            relationships_with_score=relationships_with_score,
            top_k=top_k,
        )

    def _rerank_relationships(
        self,
        relationships_with_score: List[Tuple[R, float]],
        top_k: int = 10,
    ) -> List[Tuple[R, float]]:
        """
        Rerank the relationship based on distance and weight
        """
        # TODO: the degree can br pre-calc and stored in the database in advanced.
        if self.with_degree:
            entity_ids = set()
            for r, _ in relationships_with_score:
                entity_ids.add(r.source_entity_id)
                entity_ids.add(r.target_entity_id)
            entity_degrees = self._kg_store.bulk_calc_entities_degrees(entity_ids)
        else:
            entity_degrees = defaultdict(EntityDegree)

        reranked_relationships = []
        for r, similarity_score in relationships_with_score:
            embedding_distance = 1 - similarity_score
            source_in_degree = entity_degrees[r.source_entity_id].in_degree
            target_out_degree = entity_degrees[r.target_entity_id].out_degree
            final_score = self._calc_relationship_weighted_score(
                embedding_distance,
                r.weight,
                source_in_degree,
                target_out_degree,
            )
            reranked_relationships.append((r, final_score))

        # Rerank relationships based on the calculated score.
        reranked_relationships.sort(key=lambda x: x[1], reverse=True)
        return reranked_relationships[:top_k]

    def _calc_relationship_weighted_score(
        self, embedding_distance: float, weight: int, in_degree: int, out_degree: int
    ) -> float:
        weighted_score = self._calc_weight_score(weight)
        degree_score = 0
        if self.with_degree:
            degree_score = self._calc_degree_score(in_degree, out_degree)
        return self.alpha * (1 / embedding_distance) + weighted_score + degree_score

    def _calc_weight_score(self, weight: float) -> float:
        weight_score = 0.0
        remaining_weight = weight

        for weight_range, coefficient in self.weight_coefficients:
            if remaining_weight <= 0:
                break
            lower_bound, upper_bound = weight_range
            applicable_weight = min(upper_bound - lower_bound, remaining_weight)
            weight_score += applicable_weight * coefficient
            remaining_weight -= applicable_weight

        return weight_score

    def _calc_degree_score(self, in_degree: int, out_degree: int) -> float:
        return (in_degree - out_degree) * self.degree_coefficient
