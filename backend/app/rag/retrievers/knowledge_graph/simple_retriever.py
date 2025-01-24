from typing import Optional, List

from sqlmodel import Session
from llama_index.core import QueryBundle
from llama_index.core.callbacks import CallbackManager
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
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
        self.kb = knowledge_base_repo.must_get(db_session, knowledge_base_id)
        self.embed_model = get_kb_embed_model(db_session, self.kb)
        self.embed_model.callback_manager = callback_manager
        self.entity_db_model = get_kb_entity_model(self.kb)
        self.relationship_db_model = get_kb_relationship_model(self.kb)
        # TODO: remove it
        dspy_lm = get_kb_dspy_llm(db_session, self.kb)
        self._kg_store = TiDBGraphStore(
            dspy_lm=dspy_lm,
            session=db_session,
            embed_model=self.embed_model,
            entity_db_model=self.entity_db_model,
            relationship_db_model=self.relationship_db_model,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        entities, relationships = self._kg_store.retrieve_with_weight(
            query_bundle.query_str,
            embedding=[],
            depth=self.config.depth,
            include_meta=self.config.include_meta,
            with_degree=self.config.with_degree,
            relationship_meta_filters=self.config.metadata_filters
            if self.config.metadata_filters
            else None,
        )
        return [
            NodeWithScore(
                node=KnowledgeGraphNode(
                    query=query_bundle.query_str,
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
        node = nodes_with_score[0].node
        return KnowledgeGraphRetrievalResult(
            query=node.query,
            entities=node.entities,
            relationships=node.relationships,
            subqueries=[],
        )
