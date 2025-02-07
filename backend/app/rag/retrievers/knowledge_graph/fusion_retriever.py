import logging

from sqlmodel import Session
from typing import List, Optional, Dict, Tuple
from llama_index.core import QueryBundle
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import ToolMetadata

from app.models import KnowledgeBase
from app.rag.retrievers.multiple_knowledge_base import MultiKBFusionRetriever
from app.rag.knowledge_base.selector import KBSelectMode
from app.rag.retrievers.knowledge_graph.simple_retriever import (
    KnowledgeGraphSimpleRetriever,
)
from app.rag.retrievers.knowledge_graph.schema import (
    KnowledgeGraphRetrieverConfig,
    KnowledgeGraphRetrievalResult,
    KnowledgeGraphNode,
    KnowledgeGraphRetriever,
)
from app.repositories import knowledge_base_repo


logger = logging.getLogger(__name__)


class KnowledgeGraphFusionRetriever(MultiKBFusionRetriever, KnowledgeGraphRetriever):
    knowledge_base_map: Dict[int, KnowledgeBase] = {}

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
        self.use_query_decompose = use_query_decompose

        # Prepare knowledge graph retrievers for knowledge bases.
        retrievers = []
        retriever_choices = []
        knowledge_bases = knowledge_base_repo.get_by_ids(db_session, knowledge_base_ids)
        self.knowledge_bases = knowledge_bases
        for kb in knowledge_bases:
            self.knowledge_base_map[kb.id] = kb
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

    def _fusion(
        self, query: str, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        return self._knowledge_graph_fusion(query, results)

    def _knowledge_graph_fusion(
        self, query: str, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        merged_entities = set()
        merged_relationships = {}
        merged_knowledge_base_ids = set()
        merged_children_nodes = []

        for nodes_with_scores in results.values():
            if len(nodes_with_scores) == 0:
                continue
            node: KnowledgeGraphNode = nodes_with_scores[0].node  # type:ignore

            # Merge knowledge base id.
            merged_knowledge_base_ids.add(node.knowledge_base_id)

            # Merge entities.
            merged_entities.update(node.entities)

            # Merge relationships.
            for r in node.relationships:
                key = r.rag_description
                if key not in merged_relationships:
                    merged_relationships[key] = r
                else:
                    merged_relationships[key].weight += r.weight
            # Merge to children nodes.
            merged_children_nodes.append(node)

        return [
            NodeWithScore(
                node=KnowledgeGraphNode(
                    query=query,
                    entities=list(merged_entities),
                    relationships=list(merged_relationships.values()),
                    knowledge_base_ids=merged_knowledge_base_ids,
                    children=merged_children_nodes,
                ),
                score=1,
            )
        ]
