import dspy

from autoflow.knowledge_graph.extractors.base import KGExtractor
from autoflow.knowledge_graph.programs.extract_covariates import (
    EntityCovariateExtractor,
)
from autoflow.knowledge_graph.programs.extract_graph import KnowledgeGraphExtractor
from autoflow.knowledge_graph.types import GeneratedKnowledgeGraph


class SimpleKGExtractor(KGExtractor):
    def __init__(self, dspy_lm: dspy.LM):
        super().__init__()
        self._dspy_lm = dspy_lm
        self._graph_extractor = KnowledgeGraphExtractor(dspy_lm)
        self._entity_metadata_extractor = EntityCovariateExtractor(dspy_lm)

    def extract(self, text: str) -> GeneratedKnowledgeGraph:
        knowledge_graph = self._graph_extractor.forward(text)
        knowledge_graph.entities = self._entity_metadata_extractor.forward(
            text, knowledge_graph.entities
        )
        return knowledge_graph
