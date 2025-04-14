from abc import abstractmethod

from autoflow.types import BaseComponent
from autoflow.knowledge_graph.types import GeneratedKnowledgeGraph


class KGExtractor(BaseComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def extract(self, text: str) -> GeneratedKnowledgeGraph:
        raise NotImplementedError()
