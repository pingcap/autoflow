import logging
from typing import Optional

import dspy

from autoflow.knowledge_graph.extractors.simple import SimpleKGExtractor
from autoflow.knowledge_graph.retrievers.weighted import WeightedGraphRetriever
from autoflow.knowledge_graph.types import (
    RetrievedKnowledgeGraph,
)
from autoflow.models.embedding_models import EmbeddingModel
from autoflow.storage.doc_store.types import Chunk
from autoflow.storage.graph_store.base import GraphStore
from autoflow.storage.graph_store.types import KnowledgeGraph
from autoflow.types import BaseComponent


logger = logging.getLogger(__name__)


class KnowledgeGraphIndex(BaseComponent):
    def __init__(
        self,
        kg_store: GraphStore,
        dspy_lm: dspy.LM,
        embedding_model: EmbeddingModel,
    ):
        super().__init__()
        self._kg_store = kg_store
        self._dspy_lm = dspy_lm
        self._embedding_model = embedding_model
        self._kg_extractor = SimpleKGExtractor(self._dspy_lm)

    def add_text(self, text: str) -> Optional[KnowledgeGraph]:
        knowledge_graph = self._kg_extractor.extract(text)
        return self._kg_store.add(knowledge_graph.to_create())

    def add_chunk(self, chunk: Chunk) -> Optional[KnowledgeGraph]:
        # Check if the chunk has been added.
        exists_relationships = self._kg_store.list_relationships(chunk_id=chunk.id)
        if len(exists_relationships) > 0:
            logger.warning(
                "The subgraph of chunk %s has already been added, skip.", chunk.id
            )
            return None

        logger.info("Extracting knowledge graph from chunk %s", chunk.id)
        knowledge_graph = self._kg_extractor.extract(chunk)
        logger.info("Knowledge graph extracted from chunk %s", chunk.id)

        return self._kg_store.add(knowledge_graph.to_create())

    def retrieve(
        self,
        query: str,
        depth: int = 2,
        metadata_filters: Optional[dict] = None,
        **kwargs,
    ) -> RetrievedKnowledgeGraph:
        retriever = WeightedGraphRetriever(
            self._kg_store,
            self._embedding_model,
            **kwargs,
        )
        return retriever.retrieve(
            query=query,
            depth=depth,
            metadata_filters=metadata_filters,
        )
