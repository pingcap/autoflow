import logging
from typing import List, Optional, Type

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TransformComponent

from sqlmodel import Session

from app.models.document import ContentFormat
from app.models.knowledge_base import (
    AdvancedChunkingConfig,
    AutoChunkingConfig,
    ChunkSplitter,
    ChunkingConfig,
    ChunkingMode,
    KnowledgeBase,
    MarkdownSplitterConfig,
    SentenceSplitterConfig,
)
from app.rag.knowledge_base.index_store import (
    get_kb_tidb_vector_store,
    get_kb_tidb_graph_store,
)
from app.rag.indices.knowledge_graph import KnowledgeGraphIndex
from app.models import Document, Chunk
from app.rag.node_parser.file.markdown import MarkdownNodeParser
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

    # TODO: move to ./indices/vector_search
    def build_vector_index_for_document(
        self, session: Session, db_document: Type[Document]
    ):
        """
        Build vector index and graph index from document.

        Build vector index will do the following:
        1. Parse document into nodes.
        2. Extract metadata from nodes by applying transformations.
        3. embedding text nodes.
        4. Insert nodes into `chunks` table.
        """
        vector_store = get_kb_tidb_vector_store(session, self._knowledge_base)
        transformations = self._get_transformations(db_document)
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self._embed_model,
            transformations=transformations,
        )

        llama_document = db_document.to_llama_document()
        logger.info(f"Start building vector index for document #{db_document.id}.")
        vector_index.insert(llama_document, source_uri=db_document.source_uri)
        logger.info(f"Finish building vector index for document #{db_document.id}.")
        vector_store.close_session()

        return

    def _get_transformations(
        self, db_document: Type[Document]
    ) -> List[TransformComponent]:
        transformations = []

        chunking_config_dict = self._knowledge_base.chunking_config
        config = ChunkingConfig.model_validate(chunking_config_dict)
        if config.mode != ChunkingMode.AUTO:
            auto_chunking_config = AutoChunkingConfig.model_validate(
                chunking_config_dict
            )
            chunking_config = AdvancedChunkingConfig(
                mode=ChunkingMode.ADVANCED,
                rules={
                    ContentFormat.TEXT: SentenceSplitterConfig(
                        chunk_size=auto_chunking_config.chunk_size,
                        chunk_overlap=auto_chunking_config.chunk_overlap,
                    ),
                    ContentFormat.MARKDOWN: MarkdownSplitterConfig(
                        chunk_size=auto_chunking_config.chunk_size,
                    ),
                },
            )
        elif config.mode == ChunkingMode.ADVANCED:
            chunking_config = AdvancedChunkingConfig.model_validate(
                chunking_config_dict
            )

        # Chunking
        content_format = db_document.content_format
        if content_format in chunking_config.rules:
            splitter_config = chunking_config.rules[content_format]
        else:
            splitter_config = chunking_config.rules[ContentFormat.TEXT]

        match splitter_config.type:
            case ChunkSplitter.MARKDOWN_SPLITTER:
                transformations.append(
                    MarkdownNodeParser(**splitter_config.model_dump(exclude={"type"}))
                )
            case ChunkSplitter.SENTENCE_SPLITTER:
                transformations.append(
                    SentenceSplitter(**splitter_config.model_dump(exclude={"type"}))
                )
            case _:
                raise ValueError(f"Unsupported splitter type: {splitter_config.type}")

        return transformations

    # TODO: move to ./indices/knowledge_graph
    def build_kg_index_for_chunk(self, session: Session, db_chunk: Type[Chunk]):
        """Build knowledge graph index from chunk.

        Build knowledge graph index will do the following:
        1. load TextNode from `chunks` table.
        2. extract entities and relations from TextNode.
        3. insert entities and relations into `entities` and `relations` table.
        """

        graph_store = get_kb_tidb_graph_store(session, self._knowledge_base)
        graph_index: KnowledgeGraphIndex = KnowledgeGraphIndex.from_existing(
            dspy_lm=self._dspy_lm,
            kg_store=graph_store,
        )

        node = db_chunk.to_llama_text_node()
        logger.info(f"Start building knowledge graph index for chunk #{db_chunk.id}.")
        graph_index.insert_nodes([node])
        logger.info(f"Finish building knowledge graph index for chunk #{db_chunk.id}.")
        graph_store.close_session()

        return
