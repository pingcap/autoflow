from ast import Dict
from typing import List, Optional
from pydantic import PrivateAttr

from llama_index.core import QueryBundle
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.indices.vector_store.retrievers.retriever import VectorStoreQuery
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from sqlmodel import Session


from app.rag.knowledge.chunk.helpers import map_nodes_to_chunks
from app.repositories import knowledge_base_repo, document_repo
from app.rag.knowledge.base import KnowledgeBase
from app.rag.postprocessors.metadata_post_filter import MetadataPostFilter
from app.rag.rerankers.resolver import resolve_reranker_by_id
from app.rag.tools.query_decomposer import QueryDecomposer
from app.rag.knowledge.chunk.schema import (
    ChunksRetrievalResult,
    KBChunkRetrieverConfig,
    RetrievedChunkDocument,
    VectorSearchRetrieverConfig,
)


class NodeMergeMap:
    _nodes: Dict[str, NodeWithScore] = PrivateAttr()

    def __init__(self):
        self._nodes = {}

    def add_node(self, node: NodeWithScore):
        if node.node.hash not in self._nodes:
            self._nodes[node.node.hash] = node
        else:
            # Keep highest score if node already exists
            self._nodes[node.node.hash].score = max(
                node.score or 0.0, self._nodes[node.node.hash].score or 0.0
            )

    def add_nodes(self, nodes: List[NodeWithScore]):
        for node in nodes:
            self.add_node(node)

    def get_nodes(self) -> List[NodeWithScore]:
        return list(self._nodes.values())


class KBChunkRetriever(BaseRetriever):
    _db_session: Session = PrivateAttr()
    _knowledge_bases: List[KnowledgeBase] = PrivateAttr()
    _query_decomposer: Optional[QueryDecomposer] = PrivateAttr()
    _node_postprocessors: Optional[BaseNodePostprocessor] = PrivateAttr()
    _num_candidates: int = PrivateAttr()
    _similarity_top_k: Optional[int] = PrivateAttr()
    _top_k: int = PrivateAttr()
    _full_document: bool = PrivateAttr()

    def __init__(
        self,
        db_session: Session,
        knowledge_bases: List[KnowledgeBase],
        query_decomposer: Optional[QueryDecomposer] = None,
        node_postprocessors: Optional[BaseNodePostprocessor] = None,
        num_candidates: int = 10,
        similarity_top_k: Optional[int] = None,
        top_k: int = 10,
        full_document: bool = False,
        callback_manager: Optional[CallbackManager] = CallbackManager([]),
    ):
        super().__init__(callback_manager=callback_manager)
        self._db_session = db_session
        self._knowledge_bases = knowledge_bases
        self._query_decomposer = query_decomposer
        self._node_postprocessors = node_postprocessors
        self._num_candidates = num_candidates
        self._similarity_top_k = similarity_top_k
        self._top_k = top_k
        self._callback_manager = callback_manager
        self._full_document = full_document

    @classmethod
    def from_config(
        cls,
        config: KBChunkRetrieverConfig,
        db_session: Session,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "KBChunkRetriever":
        knowledge_bases = knowledge_base_repo.get_by_ids(
            db_session, config.knowledge_base_ids
        )

        # Initialize query decomposer
        query_decomposer = None
        if config.query_decomposer:
            query_decomposer = QueryDecomposer(
                db_session=db_session,
                model_id=config.query_decomposer.llm_id,
            )

        # Initialize postprocessors
        node_postprocessors = []
        filter_config = config.metadata_filter
        if filter_config and filter_config.enabled:
            metadata_filter = MetadataPostFilter(filter_config.filters)
            node_postprocessors.append(metadata_filter)

        reranker_config = config.reranker
        if reranker_config and reranker_config.enabled:
            reranker = resolve_reranker_by_id(
                db_session, reranker_config.reranker_id, reranker_config.top_n
            )
            node_postprocessors.append(reranker)

        return cls(
            db_session=db_session,
            knowledge_bases=knowledge_bases,
            query_decomposer=query_decomposer,
            node_postprocessors=node_postprocessors,
            num_candidates=config.num_candidates,
            similarity_top_k=config.similarity_top_k,
            top_k=config.top_k,
            full_document=config.full_document,
            callback_manager=callback_manager,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Query decomposition.
        queries = [query_bundle]
        if self._query_decomposer:
            subquestions = self._query_decomposer.decompose(query_bundle.query_str)
            queries = [QueryBundle(r.question) for r in subquestions]

        all_nodes = NodeMergeMap()
        for query in queries:
            found_nodes = []

            # Retrieve chunk nodes from knowledge bases.
            for kb in self._knowledge_bases:
                if query.embedding is None and len(query.embedding_strs) > 0:
                    query.embedding = kb.embed_model.get_agg_embedding_from_queries(
                        query.embedding_strs
                    )

                result = kb.vector_store.query(
                    VectorStoreQuery(
                        query_str=query.query_str,
                        query_embedding=query.embedding,
                        similarity_top_k=self.similarity_top_k or self.top_k,
                    )
                )
                found_nodes = self._build_node_list_from_query_result(result)

            # Apply postprocessors to rerank or filter the found nodes.
            for postprocessor in self._node_postprocessors:
                found_nodes = postprocessor.postprocess_nodes(
                    found_nodes, query_bundle=query
                )

            # Merge nodes with their scores to NodeMergeMap.
            all_nodes.add_nodes(found_nodes)

        # Sort by score and return top k.
        sorted_nodes = sorted(
            all_nodes.get_nodes(), key=lambda x: x.score or 0.0, reverse=True
        )
        return sorted_nodes[: self.top_k]

    def _build_node_list_from_query_result(
        self, query_result: VectorStoreQueryResult
    ) -> List[NodeWithScore]:
        node_with_scores: List[NodeWithScore] = []
        for ind, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]
            node_with_scores.append(NodeWithScore(node=node, score=score))
        return node_with_scores

    def retrieve_chunks(self, query_str: str) -> ChunksRetrievalResult:
        nodes_with_score = self._retrieve(QueryBundle(query_str))
        chunks = map_nodes_to_chunks(nodes_with_score)

        document_ids = [c.document_id for c in chunks]
        documents = document_repo.fetch_by_ids(self._db_session, document_ids)
        if self._full_document:
            return ChunksRetrievalResult(chunks=chunks, documents=documents)
        else:
            return ChunksRetrievalResult(
                chunks=chunks,
                documents=[
                    RetrievedChunkDocument(
                        id=d.id, name=d.name, source_uri=d.source_uri
                    )
                    for d in documents
                ],
            )
