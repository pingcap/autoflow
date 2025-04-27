import logging

from pydantic import PrivateAttr
from sqlmodel import Session
from typing import List, Optional
from llama_index.core import QueryBundle
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import NodeWithScore

from app.models import KnowledgeBase
from app.models.entity import EntityType
from app.rag.knowledge.graph.schema import MetadataFilterConfig, RetrievedEntity, RetrievedRelationship
from app.rag.knowledge.graph.schema import (
    KnowledgeGraphRetrieverConfig,
    KnowledgeGraphRetrievalResult,
    KnowledgeGraphNode,
    KnowledgeGraphRetriever,
)
from app.rag.storage.graph_stores.helpers import DEFAULT_RANGE_SEARCH_CONFIG, get_query_embedding
from app.rag.tools.query_decomposer.decomposer import QueryDecomposer
from app.repositories import knowledge_base_repo


logger = logging.getLogger(__name__)


class KBKnowledgeGraphRetriever(KnowledgeGraphRetriever):
    _db_session: Session = PrivateAttr()
    _depth: int = PrivateAttr()
    _include_meta: bool = PrivateAttr()
    _with_degree: bool = PrivateAttr()
    _metadata_filter: Optional[MetadataFilterConfig] = PrivateAttr()

    def __init__(
        self,
        db_session: Session,
        knowledge_bases: List[KnowledgeBase],
        query_decomposer: Optional[QueryDecomposer] = None,
        depth: int = 2,
        include_meta: bool = False,
        with_degree: bool = False,
        metadata_filter: Optional[MetadataFilterConfig] = None,
        callback_manager: Optional[CallbackManager] = CallbackManager([]),
        **kwargs,
    ):
        super().__init__(
            callback_manager=callback_manager,
            **kwargs,
        )
        self._db_session = db_session
        self._knowledge_bases = knowledge_bases
        self._query_decomposer = query_decomposer
        self._depth = depth
        self._include_meta = include_meta
        self._with_degree = with_degree
        self._metadata_filter = metadata_filter
        self._callback_manager = callback_manager

    @classmethod
    def from_config(
        cls,
        config: KnowledgeGraphRetrieverConfig,
        db_session: Session,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "KBKnowledgeGraphRetriever":
        knowledge_bases = knowledge_base_repo.get_by_ids(db_session, config.knowledge_base_ids)

        # Initialize query decomposer
        query_decomposer = None
        if config.query_decomposer:
            query_decomposer = QueryDecomposer(
                db_session=db_session,
                model_id=config.query_decomposer.llm_id,
            )

        return cls(
            db_session=db_session,
            knowledge_bases=knowledge_bases,
            query_decomposer=query_decomposer,
            depth=config.depth,
            include_meta=config.include_meta,
            with_degree=config.with_degree,
            metadata_filter=config.metadata_filter,
            callback_manager=callback_manager,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Query decomposition.
        queries = [query_bundle]
        if self._query_decomposer:
            subquestions = self._query_decomposer.decompose(query_bundle.query_str)
            queries = [QueryBundle(r.question) for r in subquestions]

        # Retrieve knowledge graph.
        results = {}
        for query in queries:

            for kb in self._knowledge_bases:
            results[query.query_str] = self._retrieve_knowledge_graph(query)

        # Fusion.
        return self._knowledge_graph_fusion(query_bundle.query_str, results)

    def retrieve_knowledge_graph(self, query_text: str) -> KnowledgeGraphRetrievalResult:
        nodes_with_score = self._retrieve(QueryBundle(query_text))
        if len(nodes_with_score) == 0:
            return KnowledgeGraphRetrievalResult()
        node: KnowledgeGraphNode = nodes_with_score[0].node  # type:ignore

        return KnowledgeGraphRetrievalResult(
            query=node.query,
            knowledge_bases=[kb.to_descriptor() for kb in self.knowledge_bases],
            entities=node.entities,
            relationships=node.relationships,
            subgraphs=[
                KnowledgeGraphRetrievalResult(
                    query=child_node.query,
                    knowledge_base=self.knowledge_base_map[
                        child_node.knowledge_base_id
                    ].to_descriptor(),
                    entities=child_node.entities,
                    relationships=child_node.relationships,
                )
                for child_node in node.children
            ],
        )
