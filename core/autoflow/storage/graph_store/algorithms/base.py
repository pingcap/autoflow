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
        return hash(self.entity.id)


class RelationshipWithScore(Generic[R]):
    relationship: R
    score: float

    def __init__(self, relationship: R, score: float):
        self.relationship = relationship
        self.score = score

    @property
    def id(self) -> int:
        return self.relationship.id

    def __hash__(self) -> int:
        return hash(self.relationship.id)


class KnowledgeGraphRetriever(ABC, Generic[E, R]):
    def __init__(self, knowledge_graph_store: "KnowledgeGraphStore"):
        self._kg_store = knowledge_graph_store

    @abstractmethod
    def search(
        self,
        query: QueryBundle,
        depth: int = 2,
        meta_filters: dict = None,
    ) -> Tuple[List[RelationshipWithScore[R]], List[E]]:
        raise NotImplementedError
