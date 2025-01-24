from typing import List, Optional, Dict, Tuple
from llama_index.core import QueryBundle
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import ToolMetadata
from sqlmodel import Session
from app.rag.retrievers.chunk.simple_retriever import (
    ChunkSimpleRetriever,
)
from app.rag.retrievers.chunk.schema import (
    VectorSearchRetrieverConfig,
    ChunksRetrievalResult,
    ChunkRetriever,
)
from app.rag.retrievers.chunk.helpers import map_nodes_to_chunks
from app.rag.retrievers.multiple_knowledge_base import MultiKBFusionRetriever
from app.rag.knowledge_base.selector import KBSelectMode
from app.repositories import knowledge_base_repo, document_repo


class ChunkFusionRetriever(MultiKBFusionRetriever, ChunkRetriever):
    def __init__(
        self,
        db_session: Session,
        knowledge_base_ids: List[int],
        llm: LLM,
        use_query_decompose: bool = False,
        kb_select_mode: KBSelectMode = KBSelectMode.ALL,
        use_async: bool = True,
        config: VectorSearchRetrieverConfig = VectorSearchRetrieverConfig(),
        callback_manager: Optional[CallbackManager] = CallbackManager([]),
        **kwargs,
    ):
        # Prepare vector search retrievers for knowledge bases.
        retrievers = []
        retriever_choices = []
        knowledge_bases = knowledge_base_repo.get_by_ids(db_session, knowledge_base_ids)
        for kb in knowledge_bases:
            retrievers.append(
                ChunkSimpleRetriever(
                    knowledge_base_id=kb.id,
                    config=config,
                    callback_manager=callback_manager,
                    db_session=db_session,
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

    def _fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        return self._simple_fusion(results)

    def _simple_fusion(self, results: Dict[Tuple[str, int], List[NodeWithScore]]):
        """Apply simple fusion."""
        # Use a dict to de-duplicate nodes
        all_nodes: Dict[str, NodeWithScore] = {}
        for nodes_with_scores in results.values():
            for node_with_score in nodes_with_scores:
                hash = node_with_score.node.hash
                if hash in all_nodes:
                    max_score = max(
                        node_with_score.score or 0.0, all_nodes[hash].score or 0.0
                    )
                    all_nodes[hash].score = max_score
                else:
                    all_nodes[hash] = node_with_score

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)

    def retrieve_chunks(
        self,
        query_str: str,
        full_document: bool = False,
    ) -> ChunksRetrievalResult:
        nodes_with_score = self._retrieve(QueryBundle(query_str))
        chunks = map_nodes_to_chunks(nodes_with_score)

        document_ids = [c.document_id for c in chunks]
        if full_document:
            documents = document_repo.list_full_documents_by_ids(
                self._db_session, document_ids
            )
        else:
            documents = document_repo.list_simple_documents_by_ids(
                self._db_session, document_ids
            )

        return ChunksRetrievalResult(chunks=chunks, documents=documents)
