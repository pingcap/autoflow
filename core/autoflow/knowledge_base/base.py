import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TransformComponent
from pydantic import PrivateAttr
from sqlalchemy import Engine
from sqlalchemy.orm.decl_api import RegistryType
from sqlmodel.main import default_registry, Field

from autoflow.datasources import DataSource
from autoflow.datasources.mime_types import SupportedMimeTypes
from autoflow.indices.knowledge_graph.base import KnowledgeGraphIndex
from autoflow.indices.knowledge_graph.extractor import KnowledgeGraphExtractor
from autoflow.indices.vector_search.base import VectorSearchIndex
from autoflow.node_parser.file.markdown import MarkdownNodeParser
from autoflow.schema import DataSourceKind, IndexMethod, BaseComponent
from autoflow.db_models.chunk import get_chunk_model
from autoflow.db_models.entity import get_entity_model
from autoflow.db_models.relationship import get_relationship_model
from autoflow.knowledge_base.config import (
    LLMConfig,
    EmbeddingModelConfig,
    ChunkingMode,
    GeneralChunkingConfig,
    ChunkSplitterConfig,
    ChunkSplitter,
    SentenceSplitterOptions,
    MarkdownNodeParserOptions,
    ChunkingConfig,
)
from autoflow.db_models.document import Document
from autoflow.knowledge_base.datasource import get_datasource_by_kind
from autoflow.models import default_model_manager, ModelManager
from autoflow.stores import TiDBDocumentStore, TiDBKnowledgeGraphStore
from autoflow.stores.document_store.base import (
    DocumentSearchResult,
    DocumentSearchQuery,
)
from autoflow.utils.dspy_lm import get_dspy_lm_by_llm


class KnowledgeBase(BaseComponent):
    _registry: RegistryType = PrivateAttr()

    id: uuid.UUID
    name: str = Field()
    index_methods: List[IndexMethod]
    description: Optional[str] = Field(default=None)
    chunking_config: Optional[ChunkingConfig] = Field(
        default_factory=GeneralChunkingConfig
    )
    llm_config: LLMConfig = Field()
    embedding_model_config: EmbeddingModelConfig = Field()
    data_sources: List[DataSource] = Field(default_factory=list)

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        index_methods: Optional[List[IndexMethod]] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        llm: LLMConfig = None,
        embedding_model: EmbeddingModelConfig = None,
        db_engine: Engine = None,
        model_manager: Optional[ModelManager] = None,
        kb_id: Optional[uuid.UUID] = None,
    ):
        super().__init__(
            id=kb_id or uuid.uuid4(),
            name=name,
            description=description,
            index_methods=index_methods or [IndexMethod.VECTOR_SEARCH],
            chunking_config=chunking_config or GeneralChunkingConfig(),
            llm_config=llm,
            embedding_model_config=embedding_model,
        )
        self._db_engine = db_engine
        self._model_manager = model_manager or default_model_manager
        self._llm = self._model_manager.resolve_llm(llm)
        self._dspy_lm = get_dspy_lm_by_llm(self._llm)
        self._graph_extractor = KnowledgeGraphExtractor(dspy_lm=self._dspy_lm)
        self._embedding_model = self._model_manager.resolve_embedding_model(
            embedding_model
        )
        self._init_stores()
        self._vector_search_index = VectorSearchIndex(
            doc_store=self._doc_store,
        )
        self._knowledge_graph_index = KnowledgeGraphIndex(
            dspy_lm=self._dspy_lm, kg_store=self._kg_store
        )

    def _init_stores(self):
        namespace_id = f"{self.id}"
        vector_dimension = self._embedding_model.dimensions

        self._registry = RegistryType(
            metadata=default_registry.metadata,
            class_registry=default_registry._class_registry.copy(),
        )

        # Init chunk table.
        document_table_name = "documents"
        chunk_table_name = f"chunks_{namespace_id}"
        self._chunk_db_model = get_chunk_model(
            chunk_table_name,
            vector_dimension=vector_dimension,
            document_table_name=document_table_name,
            document_db_model=Document,
            registry=self._registry,
        )

        # Init entity table.
        entity_table_name = f"entities_{namespace_id}"
        self._entity_db_model = get_entity_model(
            entity_table_name,
            vector_dimension=vector_dimension,
            registry=self._registry,
        )

        # Init relationship table.
        relationship_table_name = f"relationships_{namespace_id}"
        self._relationship_db_model = get_relationship_model(
            relationship_table_name,
            vector_dimension=vector_dimension,
            entity_db_model=self._entity_db_model,
            registry=self._registry,
        )

        self._doc_store = TiDBDocumentStore[Document, self._chunk_db_model](
            db_engine=self._db_engine,
            embedding_model=self._embedding_model,
            document_db_model=Document,
            chunk_db_model=self._chunk_db_model,
        )
        self._doc_store.ensure_table_schema()

        self._kg_store = TiDBKnowledgeGraphStore(
            db_engine=self._db_engine,
            embedding_model=self._embedding_model,
            entity_db_model=self._entity_db_model,
            relationship_db_model=self._relationship_db_model,
        )
        self._kg_store.ensure_table_schema()

    def import_documents_from_datasource(
        self,
        kind: DataSourceKind,
        config: Dict[str, Any] = None,
        # TODO: Metadata Extractor
    ) -> DataSource:
        datasource = get_datasource_by_kind(kind, config)
        for doc in datasource.load_documents():
            doc.data_source_id = datasource.id
            doc.knowledge_base_id = self.id
            self.add_document(doc)
            self.build_index_for_document(doc)
        return datasource

    def import_documents_from_files(self, files: List[Path]) -> List[Document]:
        datasource = get_datasource_by_kind(
            DataSourceKind.FILE, {"files": [{"path": file.as_uri()} for file in files]}
        )
        documents = []
        for doc in datasource.load_documents():
            self.add_document(doc)
            self.build_index_for_document(doc)
        return documents

    def build_index_for_document(self, doc: Document):
        # Chunking
        chunks = self._chunking(doc)

        # Build Vector Search Index.
        if IndexMethod.VECTOR_SEARCH in self.index_methods:
            self._vector_search_index.build_index_for_chunks(chunks)

        # Build Knowledge Graph Index.
        if IndexMethod.KNOWLEDGE_GRAPH in self.index_methods:
            self._knowledge_graph_index.build_index_for_chunks(chunks)

    def _chunking(self, doc: Document):
        text_splitter = self._get_text_splitter(doc)
        nodes = text_splitter.get_nodes_from_documents([doc.to_llama_document()])
        return [
            self._chunk_db_model(
                hash=node.hash,
                text=node.text,
                meta={},
                document_id=doc.id,
            )
            for node in nodes
        ]

    def add_document(self, document: Document):
        return self._doc_store.add([document])

    def add_documents(self, documents: List[Document]):
        return self._doc_store.add(documents)

    def list_documents(self) -> List[Document]:
        return self._doc_store.list()

    def get_document(self, doc_id: int) -> Document:
        return self._doc_store.get(doc_id)

    def delete_document(self, doc_id: int) -> None:
        return self._doc_store.delete(doc_id)

    def _get_text_splitter(self, db_document: Document) -> TransformComponent:
        chunking_config = self.chunking_config
        if chunking_config.mode == ChunkingMode.ADVANCED:
            rules = chunking_config.rules
        else:
            rules = {
                SupportedMimeTypes.PLAIN_TXT: ChunkSplitterConfig(
                    splitter=ChunkSplitter.SENTENCE_SPLITTER,
                    splitter_options=SentenceSplitterOptions(
                        chunk_size=chunking_config.chunk_size,
                        chunk_overlap=chunking_config.chunk_overlap,
                        paragraph_separator=chunking_config.paragraph_separator,
                    ),
                ),
                SupportedMimeTypes.MARKDOWN: ChunkSplitterConfig(
                    splitter=ChunkSplitter.MARKDOWN_NODE_PARSER,
                    splitter_options=MarkdownNodeParserOptions(
                        chunk_size=chunking_config.chunk_size,
                    ),
                ),
            }

        # Chunking
        mime_type = db_document.mime_type
        if mime_type not in rules:
            raise RuntimeError(
                f"Can not chunking for the document in {db_document.mime_type} format"
            )

        rule = rules[mime_type]
        match rule.splitter:
            case ChunkSplitter.MARKDOWN_NODE_PARSER:
                options = MarkdownNodeParserOptions.model_validate(
                    rule.splitter_options
                )
                return MarkdownNodeParser(**options.model_dump())
            case ChunkSplitter.SENTENCE_SPLITTER:
                options = SentenceSplitterOptions.model_validate(rule.splitter_options)
                return SentenceSplitter(**options.model_dump())
            case _:
                raise ValueError(f"Unsupported chunking splitter type: {rule.splitter}")

    def search_documents(self, query: DocumentSearchQuery) -> DocumentSearchResult:
        return self._doc_store.search(query)
