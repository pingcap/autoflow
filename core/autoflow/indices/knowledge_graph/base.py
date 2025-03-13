import dspy
import logging
from typing import Any, List

import llama_index.core.instrumentation as instrument
from sqlmodel import SQLModel
from autoflow.indices.knowledge_graph.extractor import KnowledgeGraphExtractor
from autoflow.storage.graph_store import KnowledgeGraphStore
from autoflow.schema import BaseComponent

logger = logging.getLogger(__name__)

dispatcher = instrument.get_dispatcher(__name__)


class KnowledgeGraphIndex(BaseComponent):
    def __init__(
        self,
        dspy_lm: dspy.LM,
        kg_store: KnowledgeGraphStore,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._dspy_lm = dspy_lm
        self._kg_store = kg_store
        self._graph_extractor = KnowledgeGraphExtractor(dspy_lm=self._dspy_lm)

    def build_index_for_chunks(self, chunks: List[SQLModel]):
        if len(chunks) == 0:
            return chunks

        # TODO: make it faster.
        for chunk in chunks:
            pred = self._graph_extractor.forward(chunk.text)
            self._kg_store.save_knowledge_graph(pred.knowledge, chunk)
