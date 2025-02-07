import logging

from typing import List, Optional, Type

from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.utils import log_vector_store_query_result
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from sqlmodel import Session
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
import llama_index.core.instrumentation as instrument

from app.models.chunk import get_kb_chunk_model
from app.models.patch.sql_model import SQLModel
from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.rerankers.resolver import resolve_reranker_by_id
from app.rag.retrievers.chunk.schema import (
    VectorSearchRetrieverConfig,
    ChunksRetrievalResult,
    ChunkRetriever,
)
from app.rag.retrievers.chunk.helpers import map_nodes_to_chunks
from app.rag.indices.vector_search.vector_store.tidb_vector_store import TiDBVectorStore
from app.rag.postprocessors.metadata_post_filter import MetadataPostFilter
from app.repositories import knowledge_base_repo, document_repo

logger = logging.getLogger(__name__)


dispatcher = instrument.get_dispatcher(__name__)


class ChunkSimpleRetriever(BaseRetriever, ChunkRetriever):
    _chunk_model: Type[SQLModel]

    def __init__(
        self,
        knowledge_base_id: int,
        config: VectorSearchRetrieverConfig,
        db_session: Optional[Session] = None,
        callback_manager: CallbackManager = CallbackManager([]),
    ):
        super().__init__()
        if not knowledge_base_id:
            raise ValueError("Knowledge base id is required")

        self._config = config
        self._db_session = db_session
        self._kb = knowledge_base_repo.must_get(db_session, knowledge_base_id)
        self._chunk_db_model = get_kb_chunk_model(self._kb)
        self._embed_model = get_kb_embed_model(db_session, self._kb)
        self._embed_model.callback_manager = callback_manager

        # Init vector store.
        self._vector_store = TiDBVectorStore(
            session=db_session,
            chunk_db_model=self._chunk_db_model,
            oversampling_factor=config.oversampling_factor,
            callback_manager=callback_manager,
        )

        # Init node postprocessors.
        node_postprocessors = []

        # Metadata filter
        filter_config = config.metadata_filter
        if filter_config and filter_config.enabled:
            metadata_filter = MetadataPostFilter(filter_config.filters)
            node_postprocessors.append(metadata_filter)

        # Reranker
        reranker_config = config.reranker
        if reranker_config and reranker_config.enabled:
            reranker = resolve_reranker_by_id(
                db_session, reranker_config.model_id, reranker_config.top_n
            )
            node_postprocessors.append(reranker)

        self._node_postprocessors = node_postprocessors

    @dispatcher.span
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
            query_bundle.embedding = self._embed_model.get_agg_embedding_from_queries(
                query_bundle.embedding_strs
            )

        result = self._vector_store.query(
            VectorStoreQuery(
                query_str=query_bundle.query_str,
                query_embedding=query_bundle.embedding,
                similarity_top_k=self._config.similarity_top_k or self._config.top_k,
            )
        )
        nodes = self._build_node_list_from_query_result(result)

        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )

        return nodes[: self._config.top_k]

    def _build_node_list_from_query_result(
        self, query_result: VectorStoreQueryResult
    ) -> List[NodeWithScore]:
        log_vector_store_query_result(query_result)
        node_with_scores: List[NodeWithScore] = []
        for ind, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]
            node_with_scores.append(NodeWithScore(node=node, score=score))

        return node_with_scores

    def retrieve_chunks(
        self, query_str: str, full_document: bool = False
    ) -> ChunksRetrievalResult:
        nodes_with_score = self.retrieve(query_str)
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
