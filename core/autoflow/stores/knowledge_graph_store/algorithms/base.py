from abc import abstractmethod, ABC
from typing import Tuple, Generic, List

from pydantic import BaseModel

from ..base import KnowledgeGraphStore, E, R
from ...schema import QueryBundle


class EntityWithScore(BaseModel, Generic[E]):
    entity: E
    score: float

    @property
    def id(self) -> int:
        return self.entity.id

    def __hash__(self) -> int:
        return hash(self.id)


class RelationshipWithScore(BaseModel, Generic[R]):
    relationship: R
    score: float

    @property
    def id(self) -> int:
        return self.relationship.id

    def __hash__(self) -> int:
        return hash(self.id)


class GraphSearchAlgorithm(ABC, Generic[E, R]):
    @abstractmethod
    def search(
        self,
        kg_store: KnowledgeGraphStore,
        query: QueryBundle,
        depth: int = 2,
        meta_filters: dict = None,
    ) -> Tuple[List[RelationshipWithScore[R]], List[E]]:
        raise NotImplementedError
