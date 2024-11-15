import logging
from typing import Optional, Type

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceSplitter
from sqlmodel import Session

from app.models.knowledge_base import KnowledgeBase
from app.rag.knowledge_base.index_store import get_kb_tidb_vector_store, get_kb_tidb_graph_store
from app.rag.knowledge_graph import KnowledgeGraphIndex
from app.core.config import settings
from app.models import (
    Document as DBDocument,
    Chunk as DBChunk,
)
from app.utils.dspy import get_dspy_lm_by_llama_llm

logger = logging.getLogger(__name__)


class IndexService:
    """
    Service class for building RAG indexes (vector index and knowledge graph index).
    """

    def __init__(
        self,
        llm: LLM,
        embed_model: Optional[EmbedType] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        self._llm = llm
        self._dspy_lm = get_dspy_lm_by_llama_llm(llm)
        self._embed_model = embed_model
        self._knowledge_base = knowledge_base

    def build_vector_index_for_document(self, session: Session, db_document: Type[DBDocument]):
        """
        Build vector index and graph index from document.

        Build vector index will do the following:
        1. Parse document into nodes.
        2. Extract metadata from nodes by applying transformations.
        3. embedding text nodes.
        4. Insert nodes into `chunks` table.
        """

        if db_document.mime_type.lower() == "text/markdown":
            # spliter = MarkdownNodeParser()
            # TODO: FIX MarkdownNodeParser
            spliter = SentenceSplitter(
                chunk_size=settings.EMBEDDING_MAX_TOKENS,
            )
        else:
            spliter = SentenceSplitter(
                chunk_size=settings.EMBEDDING_MAX_TOKENS,
            )

        _transformations = [spliter]

        vector_store = get_kb_tidb_vector_store(session, self._knowledge_base)
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self._embed_model,
            transformations=_transformations
        )

        document = db_document.to_llama_document()
        logger.info(f"Start building vector index for document #{document.doc_id}.")
        vector_index.insert(document, source_uri=db_document.source_uri)
        logger.info(f"Finish building vector index for document #{document.doc_id}.")
        vector_store.close_session()

        return


    def build_kg_index_for_chunk(self, session: Session, db_chunk: Type[DBChunk]):
        """Build knowledge graph index from chunk.

        Build knowledge graph index will do the following:
        1. load TextNode from `chunks` table.
        2. extract entities and relations from TextNode.
        3. insert entities and relations into `entities` and `relations` table.
        """

        graph_store = get_kb_tidb_graph_store(session, self._knowledge_base)
        graph_index: KnowledgeGraphIndex = KnowledgeGraphIndex.from_existing(
            dspy_lm=self._dspy_lm, kg_store=graph_store
        )

        node = db_chunk.to_llama_text_node()
        logger.info(f"Start building knowledge graph index for chunk #{db_chunk.id}.")
        graph_index.insert_nodes([node])
        logger.info(f"Finish building knowledge graph index for chunk #{db_chunk.id}.")
        graph_store.close_session()

        return
