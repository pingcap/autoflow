import logging

from typing import List, Optional, Type

from llama_index.core.callbacks import CallbackManager
from sqlmodel import Session
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
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

        # Vector Index
        vector_store = TiDBVectorStore(
            session=db_session,
            chunk_db_model=self._chunk_db_model,
            oversampling_factor=config.oversampling_factor,
        )
        self._vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self._embed_model,
            callback_manager=callback_manager,
        )

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

        # Vector Index Retrieve Engine
        self._retrieve_engine = self._vector_index.as_retriever(
            node_postprocessors=node_postprocessors,
            similarity_top_k=config.similarity_top_k or config.top_k,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retrieve_engine.retrieve(query_bundle)
        return nodes[: self._config.top_k]

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
