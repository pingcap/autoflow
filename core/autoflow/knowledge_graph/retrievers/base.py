from abc import abstractmethod, ABC

from autoflow.storage.graph_store.base import GraphStore
from autoflow.storage.types import QueryBundle
from autoflow.knowledge_graph.types import RetrievedKnowledgeGraph


class KGRetriever(ABC):
    def __init__(self, knowledge_graph_store: GraphStore):
        self._kg_store = knowledge_graph_store

    @abstractmethod
    def retrieve(
        self,
        query: QueryBundle,
        depth: int = 2,
        meta_filters: dict = None,
    ) -> RetrievedKnowledgeGraph:
        raise NotImplementedError
