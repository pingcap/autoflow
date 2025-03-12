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
from autoflow.indices.vector_search.base import VectorSearchIndex
from autoflow.storage.doc_store.base import DocumentSearchMethod
from autoflow.transformers.markdown import MarkdownNodeParser
from autoflow.schema import DataSourceType, IndexMethod, BaseComponent
from autoflow.models.chunk import get_chunk_model
from autoflow.models.entity import get_entity_model
from autoflow.models.relationship import get_relationship_model
from autoflow.knowledge_base.config import (
    ChunkingMode,
    GeneralChunkingConfig,
    ChunkSplitterConfig,
    ChunkSplitter,
    SentenceSplitterOptions,
    MarkdownNodeParserOptions,
    ChunkingConfig,
)
from autoflow.models.document import Document
from autoflow.knowledge_base.datasource import get_datasource_by_type
from autoflow.llms import default_llm_manager, LLMManager
from autoflow.llms.chat_models import ChatModel
from autoflow.llms.embeddings import EmbeddingModel
from autoflow.storage.graph_store import TiDBKnowledgeGraphStore
from autoflow.storage.doc_store import (
    TiDBDocumentStore,
    DocumentSearchResult,
)
from autoflow.storage.graph_store.base import GraphSearchAlgorithm
from autoflow.storage.schema import QueryBundle


class KnowledgeBase(BaseComponent):
    _registry: RegistryType = PrivateAttr()

    id: uuid.UUID
    name: str = Field()
    index_methods: List[IndexMethod]
    description: Optional[str] = Field(default=None)
    chunking_config: Optional[ChunkingConfig] = Field(
        default_factory=GeneralChunkingConfig
    )
    data_sources: List[DataSource] = Field(default_factory=list)

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        index_methods: Optional[List[IndexMethod]] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        chat_model: Optional[ChatModel] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        db_engine: Engine = None,
        llm_manager: Optional[LLMManager] = None,
        id: Optional[uuid.UUID] = None,
    ):
        super().__init__(
            id=id or uuid.uuid4(),
            name=name,
            description=description,
            index_methods=index_methods or [IndexMethod.VECTOR_SEARCH],
            chunking_config=chunking_config or GeneralChunkingConfig(),
        )
        self._db_engine = db_engine
        self._model_manager = llm_manager or default_llm_manager
        self._chat_model = chat_model

        from autoflow.utils.dspy_lm import get_dspy_lm_by_chat_model

        self._dspy_lm = get_dspy_lm_by_chat_model(self._chat_model)

        self._embedding_model = embedding_model
        self._init_stores()
        self._vector_search_index = VectorSearchIndex(
            doc_store=self._doc_store,
        )

        from autoflow.indices.knowledge_graph.base import KnowledgeGraphIndex

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
        type: DataSourceType,
        config: Dict[str, Any] = None,
        # TODO: Metadata Extractor
    ) -> DataSource:
        datasource = get_datasource_by_type(type, config)
        for doc in datasource.load_documents():
            doc.data_source_id = datasource.id
            doc.knowledge_base_id = self.id
            self.add_document(doc)
            self.build_index_for_document(doc)
        return datasource

    def import_documents_from_files(self, files: List[Path]) -> List[Document]:
        datasource = get_datasource_by_type(
            DataSourceType.FILE, {"files": [{"path": file.as_uri()} for file in files]}
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

    def search_documents(
        self,
        query: str,
        search_method: Optional[List[DocumentSearchMethod]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        similarity_nprobe: Optional[int] = None,
        similarity_top_k: Optional[int] = 5,
        **kwargs: Any,
    ) -> DocumentSearchResult:
        return self._doc_store.search(
            query=query,
            search_method=search_method or [DocumentSearchMethod.VECTOR_SEARCH],
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_nprobe=similarity_nprobe,
            similarity_top_k=similarity_top_k,
            **kwargs,
        )

    def search_knowledge_graph(
        self,
        query: str,
        depth: int = 2,
        include_meta: bool = False,
        metadata_filters: Optional[dict] = None,
        search_algorithm: Optional[
            GraphSearchAlgorithm
        ] = GraphSearchAlgorithm.WEIGHTED_SEARCH,
        **kwargs,
    ):
        return self._kg_store.search(
            query=QueryBundle(query_str=query),
            depth=depth,
            include_meta=include_meta,
            metadata_filters=metadata_filters,
            search_algorithm=search_algorithm,
            **kwargs,
        )
