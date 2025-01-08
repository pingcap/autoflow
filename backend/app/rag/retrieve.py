import logging
from typing import List, Optional
from llama_index.core.schema import NodeWithScore
from sqlmodel import Session

from app.models import (
    Document as DBDocument,
)
from app.rag.retrievers.LegacyChatEngineRetriever import (
    ChatEngineBasedRetriever,
)
from app.repositories.chunk import ChunkRepo
from app.repositories.knowledge_base import knowledge_base_repo
from app.models.chunk import get_kb_chunk_model
from app.rag.chat_config import ChatEngineConfig


logger = logging.getLogger(__name__)


class ChatEngineBasedRetrieveService:
    def chat_engine_retrieve_documents(
        self,
        db_session: Session,
        question: str,
        top_k: int = 5,
        chat_engine_name: str = "default",
        similarity_top_k: Optional[int] = None,
        oversampling_factor: Optional[int] = None,
        enable_kg_enchance_query_refine: bool = False,
    ) -> List[DBDocument]:
        chat_engine_config = ChatEngineConfig.load_from_db(db_session, chat_engine_name)
        if not chat_engine_config.knowledge_base:
            raise Exception("Chat engine does not configured with knowledge base")

        nodes = self.chat_engine_retrieve_chunks(
            db_session,
            question,
            top_k,
            chat_engine_name,
            similarity_top_k,
            oversampling_factor,
            enable_kg_enchance_query_refine,
        )

        linked_knowledge_base = chat_engine_config.knowledge_base.linked_knowledge_base
        kb = knowledge_base_repo.must_get(db_session, linked_knowledge_base.id)
        chunk_model = get_kb_chunk_model(kb)
        chunk_repo = ChunkRepo(chunk_model)
        chunk_ids = [node.node.node_id for node in nodes]

        return chunk_repo.get_documents_by_chunk_ids(chunk_ids)

    def chat_engine_retrieve_chunks(
        self,
        db_session: Session,
        question: str,
        top_k: int = 5,
        chat_engine_name: str = "default",
        similarity_top_k: Optional[int] = None,
        oversampling_factor: Optional[int] = None,
        enable_kg_enchance_query_refine: bool = False,
    ) -> List[NodeWithScore]:
        retriever = ChatEngineBasedRetriever(
            db_session=db_session,
            engine_name=chat_engine_name,
            top_k=top_k,
            similarity_top_k=similarity_top_k,
            oversampling_factor=oversampling_factor,
            enable_kg_enchance_query_refine=enable_kg_enchance_query_refine,
        )
        return retriever.retrieve(question)


retrieve_service = ChatEngineBasedRetrieveService()
