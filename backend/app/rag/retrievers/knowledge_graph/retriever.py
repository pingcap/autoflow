from typing import Optional, List, Tuple

import dspy
from llama_index.core import QueryBundle
from llama_index.core.callbacks import CallbackManager
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from sqlmodel import Session

from app.core.db import engine
from app.models.entity import get_kb_entity_model
from app.models.relationship import get_kb_relationship_model
from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.knowledge_graph import KnowledgeGraphIndex
from app.rag.knowledge_graph.graph_store import TiDBGraphStore
from app.rag.retrievers.knowledge_graph.config import KnowledgeGraphRetrieverConfig
from app.repositories import knowledge_base_repo


class KnowledgeGraphRetriever(BaseRetriever):
    def __init__(
        self,
        config: KnowledgeGraphRetrieverConfig,
        llm: dspy.LM,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ):
        super().__init__(callback_manager, **kwargs)
        self.config = config
        with Session(engine) as session:
            kb = knowledge_base_repo.must_get(session, config.knowledge_base_id)
            self.embed_model = get_kb_embed_model(session, kb)
            self.entity_db_model = get_kb_entity_model(kb)
            self.relationship_db_model = get_kb_relationship_model(kb)
            self.graph_store = TiDBGraphStore(
                dspy_lm=llm,
                embed_model=self.embed_model,
                entity_db_model=self.entity_db_model,
                relationship_db_model=self.relationship_db_model,
            )
            self._graph_index = KnowledgeGraphIndex.from_existing(
                dspy_lm=llm,
                kg_store=self.graph_store,
                callback_manager=callback_manager,
            )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        pass

    def retrieve_graph(
        self, query_bundle: QueryBundle
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        return self._graph_index.retrieve_with_weight(
            query_bundle.query_str,
            embedding=[],
            depth=self.config.depth,
            include_meta=self.config.include_meta,
            with_degree=self.config.with_degree,
            relationship_meta_filters=self.config.relationship_meta_filters,
            with_chunks=True,
        )
