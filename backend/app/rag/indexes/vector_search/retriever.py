import logging

from typing import Dict, List, Optional, Type
from pydantic import BaseModel
from sqlmodel import Session, select
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.core.db import engine
from app.models.chunk import get_kb_chunk_model
from app.models.document import Document
from app.models.patch.sql_model import SQLModel
from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.rerankers.resolver import resolve_reranker_by_id
from app.rag.indexes.vector_search.config import VectorSearchConfig
from app.rag.indexes.vector_search.vector_store.tidb_vector_store import TiDBVectorStore
from app.rag.postprocessors.resolver import get_metadata_post_filter
from app.repositories import knowledge_base_repo

logger = logging.getLogger(__name__)


class RetrievedChunkDocument(BaseModel):
    id: int
    name: str
    source_uri: str


class RetrievedChunk(BaseModel):
    id: str
    text: str
    metadata: dict
    document: RetrievedChunkDocument
    score: float


class VectorSearchRetriever(BaseRetriever):
    _chunk_model: Type[SQLModel]

    def __init__(
        self,
        knowledge_base_id: int,
        config: VectorSearchConfig,
        db_session: Optional[Session] = None,
    ):
        super().__init__()
        if not knowledge_base_id:
            raise ValueError("Knowledge base id is required")

        with db_session or Session(engine) as session:
            self._kb = knowledge_base_repo.must_get(session, knowledge_base_id)
            self._chunk_db_model = get_kb_chunk_model(self._kb)
            self._embed_model = get_kb_embed_model(session, self._kb)

            # Vector Index
            vector_store = TiDBVectorStore(
                session=session,
                chunk_db_model=self._chunk_db_model,
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
                    session, config.reranker.model_id, config.reranker.top_n
                )
                node_postprocessors.append(reranker)

            # Vector Index Retrieve Engine
            self._retrieve_engine = self._vector_index.as_retriever(
                node_postprocessors=node_postprocessors,
                similarity_top_k=config.similarity_top_k or config.top_k,
            )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retrieve_engine.retrieve(query_bundle)

    def retrieve_chunks(
        self, query_bundle: QueryBundle, db_session: Optional[Session] = None
    ) -> List[RetrievedChunk]:
        nodes_with_score = self._retrieve(query_bundle)
        return self.map_nodes_to_chunks(nodes_with_score, db_session)

    def map_nodes_to_chunks(
        self, nodes_with_score, db_session: Optional[Session] = None
    ):
        chunk_ids = [ns.node.node_id for ns in nodes_with_score]
        chunk_to_document_map = self._get_chunk_to_document_map(chunk_ids, db_session)

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

    def _get_chunk_to_document_map(
        self, chunk_ids: List[str], db_session: Optional[Session] = None
    ) -> Dict[str, RetrievedChunkDocument]:
        stmt = (
            select(
                self._chunk_db_model.id,
                Document.id,
                Document.name,
                Document.source_uri,
            )
            .outerjoin(Document, self._chunk_db_model.document_id == Document.id)
            .where(
                self._chunk_db_model.id.in_(chunk_ids),
            )
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
