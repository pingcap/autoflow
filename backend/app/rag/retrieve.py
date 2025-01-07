import logging
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from sqlmodel import Session

from app.models import (
    Document as DBDocument,
)
from app.rag.chat_config import ChatEngineConfig
from app.rag.vector_store.tidb_vector_store import TiDBVectorStore
from backend.app.models.chunk import get_kb_chunk_model
from backend.app.rag.retrievers.LegacyChatEngineRetriever import (
    LegacyChatEngineRetriever,
)
from backend.app.repositories.chunk import ChunkRepo


logger = logging.getLogger(__name__)


class RetrieveService:
    def chat_engine_retrieve_documents(
        self,
        db_session: Session,
        question: str,
        top_k: int = 5,
        chat_engine_name: str = "default",
    ) -> List[DBDocument]:
        chat_engine_config = ChatEngineConfig.load_from_db(db_session, chat_engine_name)
        if chat_engine_config.knowledge_base is None:
            logger.warning(
                f"Knowledge base is not set for chat engine {chat_engine_name}"
            )
            return []

        nodes = self.chat_engine_retrieve_chunks(
            db_session, question, top_k, chat_engine_name
        )
        if not nodes:
            return []

        chunk_model = get_kb_chunk_model(chat_engine_config.knowledge_base)
        chunk_repo = ChunkRepo(chunk_model)
        source_nodes_ids = [node.node_id for node in nodes]
        return chunk_repo.get_documents_by_chunk_ids(db_session, source_nodes_ids)

    def chat_engine_retrieve_chunks(
        self,
        db_session: Session,
        question: str,
        top_k: int = 5,
        chat_engine_name: str = "default",
    ) -> List[NodeWithScore]:
        retriever = LegacyChatEngineRetriever(db_session, chat_engine_name, top_k)
        return retriever.retrieve(question)

    def retrieve_chunks(self, request: RetrieveRequest) -> List[NodeWithScore]:
        vector_store = TiDBVectorStore(
            session=self.db_session, chunk_db_model=self._chunk_model
        )
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=self._embed_model
        )
        retrieve_engine = vector_index.as_retriever(
            node_postprocessors=[self._reranker],
            similarity_top_k=top_k,
        )

        node_list: List[NodeWithScore] = retrieve_engine.retrieve(question)
        return node_list


retrieve_service = RetrieveService()
