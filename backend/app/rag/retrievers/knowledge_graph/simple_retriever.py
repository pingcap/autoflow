from typing import Optional, List

from sqlmodel import Session
from llama_index.core import QueryBundle
from llama_index.core.callbacks import CallbackManager
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore

from app.models.chunk import get_kb_chunk_model
from app.models.entity import get_kb_entity_model
from app.models.relationship import get_kb_relationship_model
from app.rag.retrievers.knowledge_graph.schema import (
    KnowledgeGraphRetrieverConfig,
    KnowledgeGraphRetrievalResult,
    KnowledgeGraphNode,
    KnowledgeGraphRetriever,
)
from app.rag.knowledge_base.config import get_kb_embed_model, get_kb_dspy_llm
from app.rag.indices.knowledge_graph.graph_store import TiDBGraphStore
from app.repositories import knowledge_base_repo


class KnowledgeGraphSimpleRetriever(BaseRetriever, KnowledgeGraphRetriever):
    def __init__(
        self,
        db_session: Session,
        knowledge_base_id: int,
        config: KnowledgeGraphRetrieverConfig,
        callback_manager: Optional[CallbackManager] = CallbackManager([]),
        **kwargs,
    ):
        super().__init__(callback_manager, **kwargs)
        self.config = config
        self._callback_manager = callback_manager
        self.knowledge_base = knowledge_base_repo.must_get(
            db_session, knowledge_base_id
        )
        self.embed_model = get_kb_embed_model(db_session, self.knowledge_base)
        self.embed_model.callback_manager = callback_manager
        self.chunk_db_model = get_kb_chunk_model(self.knowledge_base)
        self.entity_db_model = get_kb_entity_model(self.knowledge_base)
        self.relationship_db_model = get_kb_relationship_model(self.knowledge_base)
        # TODO: remove it
        dspy_lm = get_kb_dspy_llm(db_session, self.knowledge_base)
        self._kg_store = TiDBGraphStore(
            knowledge_base=self.knowledge_base,
            dspy_lm=dspy_lm,
            session=db_session,
            embed_model=self.embed_model,
            entity_db_model=self.entity_db_model,
            relationship_db_model=self.relationship_db_model,
            chunk_db_model=self.chunk_db_model,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        metadata_filters = {}
        if self.config.metadata_filter and self.config.metadata_filter.enabled:
            metadata_filters = self.config.metadata_filter.filters

        entities, relationships = self._kg_store.retrieve_with_weight(
            query_bundle.query_str,
            embedding=[],
            depth=self.config.depth,
            include_meta=self.config.include_meta,
            with_degree=self.config.with_degree,
            relationship_meta_filters=metadata_filters,
        )
        return [
            NodeWithScore(
                node=KnowledgeGraphNode(
                    query=query_bundle.query_str,
                    knowledge_base_id=self.knowledge_base.id,
                    entities=entities,
                    relationships=relationships,
                ),
                score=1,
            )
        ]

    def retrieve_knowledge_graph(
        self, query_text: str
    ) -> KnowledgeGraphRetrievalResult:
        nodes_with_score = self._retrieve(QueryBundle(query_text))
        if len(nodes_with_score) == 0:
            return KnowledgeGraphRetrievalResult()
        node: KnowledgeGraphNode = nodes_with_score[0].node  # type:ignore
        return KnowledgeGraphRetrievalResult(
            query=node.query,
            knowledge_base=self.knowledge_base.to_descriptor(),
            entities=node.entities,
            relationships=node.relationships,
            subgraphs=[],
        )
