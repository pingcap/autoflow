import logging

from sqlmodel import Session
from typing import List, Optional, Dict, Tuple
from llama_index.core import QueryBundle
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import ToolMetadata

from app.rag.knowledge_base.multi_kb_retriever import MultiKBFusionRetriever
from app.rag.knowledge_base.selector import KBSelectMode
from app.rag.indices.knowledge_graph.retriever.simple_retriever import (
    KnowledgeGraphSimpleRetriever,
)
from app.rag.indices.knowledge_graph.retriever.schema import (
    KnowledgeGraphRetrieverConfig,
    RetrievedRelationship,
    KnowledgeGraphRetrievalResult,
    KnowledgeGraphNode,
    KnowledgeGraphRetriever,
)
from app.repositories import knowledge_base_repo


logger = logging.getLogger(__name__)


class KnowledgeGraphFusionSimpleRetriever(
    MultiKBFusionRetriever, KnowledgeGraphRetriever
):
    def __init__(
        self,
        db_session: Session,
        knowledge_base_ids: List[int],
        llm: LLM,
        use_query_decompose: bool = False,
        kb_select_mode: KBSelectMode = KBSelectMode.ALL,
        use_async: bool = True,
        config: KnowledgeGraphRetrieverConfig = KnowledgeGraphRetrieverConfig(),
        callback_manager: Optional[CallbackManager] = CallbackManager([]),
        **kwargs,
    ):
        # Prepare knowledge graph retrievers for knowledge bases.
        retrievers = []
        retriever_choices = []
        knowledge_bases = knowledge_base_repo.get_by_ids(db_session, knowledge_base_ids)
        for kb in knowledge_bases:
            retrievers.append(
                KnowledgeGraphSimpleRetriever(
                    db_session=db_session,
                    knowledge_base_id=kb.id,
                    config=config,
                    callback_manager=callback_manager,
                )
            )
            retriever_choices.append(
                ToolMetadata(
                    name=kb.name,
                    description=kb.description,
                )
            )

        super().__init__(
            db_session=db_session,
            retrievers=retrievers,
            retriever_choices=retriever_choices,
            llm=llm,
            use_query_decompose=use_query_decompose,
            kb_select_mode=kb_select_mode,
            use_async=use_async,
            callback_manager=callback_manager,
            **kwargs,
        )

    def retrieve_knowledge_graph(
        self, query_text: str
    ) -> KnowledgeGraphRetrievalResult:
        nodes_with_score = self._retrieve(QueryBundle(query_text))
        if len(nodes_with_score) == 0:
            return KnowledgeGraphRetrievalResult()
        node: KnowledgeGraphNode = nodes_with_score[0].node  # type:ignore
        subqueries = [
            KnowledgeGraphRetrievalResult(
                query=subgraph.query,
                entities=subgraph.entities,
                relationships=subgraph.relationships,
            )
            for subgraph in node.subqueries.values()
        ]

        return KnowledgeGraphRetrievalResult(
            query=node.query,
            entities=node.entities,
            relationships=node.relationships,
            subqueries=[
                KnowledgeGraphRetrievalResult(
                    query=sub.query,
                    entities=sub.entities,
                    relationships=node.relationships,
                )
                for sub in subqueries
            ],
        )

    def _fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        return self._knowledge_graph_fusion(results)

    def _knowledge_graph_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        merged_queries = {}
        merged_entities = {}
        merged_relationships = {}
        for nodes_with_scores in results.values():
            if len(nodes_with_scores) == 0:
                continue
            node: KnowledgeGraphNode = nodes_with_scores[0].node  # type:ignore
            merged_queries[node.query] = node

            # Merge entities.
            for e in node.entities:
                if e.id not in merged_entities:
                    merged_entities[e.id] = e

            # Merge relationships.
            for r in node.relationships:
                key = (r.source_entity_id, r.target_entity_id, r.description)
                if key not in merged_relationships:
                    merged_relationships[key] = RetrievedRelationship(
                        id=r.id,
                        source_entity_id=r.source_entity_id,
                        target_entity_id=r.target_entity_id,
                        description=r.description,
                        rag_description=r.rag_description,
                        weight=0,
                        meta=r.meta,
                        last_modified_at=r.last_modified_at,
                    )
                else:
                    merged_relationships[key].weight += r.weight

        return [
            NodeWithScore(
                node=KnowledgeGraphNode(
                    query=None,
                    entities=list(merged_entities.values()),
                    relationships=list(merged_relationships.values()),
                    subqueries=merged_queries,
                ),
                score=1,
            )
        ]
