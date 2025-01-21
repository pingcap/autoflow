import logging

from typing import Dict, List, Optional, Type

from llama_index.core.callbacks import CallbackManager
from sqlmodel import Session, select
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from app.models.chunk import get_kb_chunk_model
from app.models.document import Document
from app.models.patch.sql_model import SQLModel
from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.rerankers.resolver import resolve_reranker_by_id
from app.rag.indices.vector_search.retriever.schema import (
    VectorSearchRetrieverConfig,
    RetrievedChunkDocument,
    RetrievedChunk,
)
from app.rag.vector_store.tidb_vector_store import TiDBVectorStore
from app.rag.postprocessors.resolver import get_metadata_post_filter
from app.repositories import knowledge_base_repo

logger = logging.getLogger(__name__)


class VectorSearchRetriever(BaseRetriever):
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
        enable_metadata_filter = config.metadata_filter is not None
        if enable_metadata_filter:
            metadata_filter = get_metadata_post_filter(config.metadata_filter.filters)
            node_postprocessors.append(metadata_filter)

        # Reranker
        enable_reranker = config.reranker is not None
        if enable_reranker:
            reranker = resolve_reranker_by_id(
                db_session, config.reranker.model_id, config.reranker.top_n
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

    def retrieve_chunks(self, query_bundle: QueryBundle) -> List[RetrievedChunk]:
        nodes_with_score = self._retrieve(query_bundle)
        return map_nodes_to_chunks(
            nodes_with_score, self._chunk_db_model, self._db_session
        )


def map_nodes_to_chunks(
    nodes_with_score, chunk_model: Type[SQLModel], db_session: Session
):
    chunk_ids = [ns.node.node_id for ns in nodes_with_score]
    chunk_to_document_map = get_chunks_reference_documents(
        chunk_ids, chunk_model, db_session
    )

    return [
        RetrievedChunk(
            id=ns.node.node_id,
            text=ns.node.text,
            metadata=ns.node.metadata,
            document=chunk_to_document_map[ns.node.node_id],
            score=ns.score,
        )
        for ns in nodes_with_score
    ]


def get_chunks_reference_documents(
    chunk_ids: List[str], chunk_model: Type[SQLModel], db_session: Session
) -> Dict[str, RetrievedChunkDocument]:
    stmt = (
        select(
            chunk_model.id,
            Document.id,
            Document.name,
            Document.source_uri,
        )
        .outerjoin(Document, chunk_model.document_id == Document.id)
        .where(chunk_model.id.in_(chunk_ids))
    )
    rows = db_session.exec(stmt).all()
    return {
        str(row[0]): RetrievedChunkDocument(
            id=row[1],
            name=row[2],
            source_uri=row[3],
        )
        for row in rows
    }
