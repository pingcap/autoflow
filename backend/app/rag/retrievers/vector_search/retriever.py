import logging

from typing import List
from sqlmodel import Session
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.core.db import engine
from app.models.chunk import get_kb_chunk_model
from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.rerankers.resolver import resolve_reranker_by_id
from app.rag.retrievers.vector_search.config import VectorSearchConfig
from app.rag.vector_store.tidb_vector_store import TiDBVectorStore
from app.rag.postprocessors.resolver import get_metadata_post_filter
from app.repositories import knowledge_base_repo

logger = logging.getLogger(__name__)


class KBVectorSearchRetriever(BaseRetriever):
    def __init__(self, config: VectorSearchConfig):
        super().__init__()
        if not config.knowledge_base_id:
            raise ValueError("Knowledge base id is required")

        with Session(engine) as session:
            self._kb = knowledge_base_repo.must_get(session, config.knowledge_base_id)
            self._chunk_model = get_kb_chunk_model(self._kb)
            self._embed_model = get_kb_embed_model(session, self._kb)

            # Vector Index
            vector_store = TiDBVectorStore(
                session=session,
                chunk_db_model=self._chunk_model,
                oversampling_factor=config.oversampling_factor,
            )
            self._vector_index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self._embed_model,
            )

            node_postprocessors = []

            # Metadata filter
            enable_metadata_filter = config.metadata_filter is not None
            if enable_metadata_filter:
                metadata_filter = get_metadata_post_filter(
                    config.metadata_filter.filters
                )
                node_postprocessors.append(metadata_filter)

            # Reranker
            enable_reranker = config.reranker is not None
            if enable_reranker:
                reranker = resolve_reranker_by_id(
                    session, config.reranker.reranker_id, config.reranker.top_n
                )
                node_postprocessors.append(reranker)

            # Vector Index Retrieve Engine
            self._retrieve_engine = self._vector_index.as_retriever(
                node_postprocessors=node_postprocessors,
                similarity_top_k=config.similarity_top_k or config.top_k,
            )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retrieve_engine.retrieve(query_bundle)
