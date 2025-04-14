import logging
from os import cpu_count
import uuid
from typing import List, Optional, Any
from functools import partial

from pydantic import Field
from sqlalchemy import Engine
from concurrent.futures import ThreadPoolExecutor

from autoflow.chunkers.base import Chunker
from autoflow.chunkers.helper import get_chunker_for_datatype
from autoflow.configs.knowledge_base import IndexMethod
from autoflow.data_types import DataType, guess_datatype
from autoflow.knowledge_graph.index import KnowledgeGraphIndex
from autoflow.loaders.base import Loader
from autoflow.loaders.helper import get_loader_for_datatype
from autoflow.models.llms import LLM
from autoflow.models.embedding_models import EmbeddingModel
from autoflow.models.llms.dspy import get_dspy_lm_by_llm
from autoflow.models.rerank_models import RerankModel
from autoflow.types import BaseComponent, SearchMode
from autoflow.storage.doc_store import DocumentSearchResult, Document

logger = logging.getLogger(__name__)


class KnowledgeBase(BaseComponent):
    namespace: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    index_methods: List[IndexMethod] = Field(default=[IndexMethod.VECTOR_SEARCH])

    def __init__(
        self,
        db_engine: Engine = None,
        namespace: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        index_methods: Optional[List[IndexMethod]] = None,
        llm: Optional[LLM] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        rerank_model: Optional[RerankModel] = None,
        max_workers: Optional[int] = None,
    ):
        super().__init__(
            namespace=namespace,
            name=name,
            description=description,
            index_methods=index_methods,
        )
        self._db_engine = db_engine
        self._llm = llm
        self._embedding_model = embedding_model
        self._reranker_model = rerank_model
        self._init_stores()
        self._init_indexes()
        self._max_workers = max_workers or cpu_count()

    def _init_stores(self):
        from autoflow.storage.doc_store.tidb_doc_store import TiDBDocumentStore
        from autoflow.storage.graph_store.tidb_graph_store import TiDBGraphStore
        from pytidb import TiDBClient

        self._tidb_client = TiDBClient(self._db_engine)
        self._doc_store = TiDBDocumentStore(
            client=self._tidb_client,
            embedding_model=self._embedding_model,
            namespace=self.namespace,
        )
        self._kg_store = TiDBGraphStore(
            client=self._tidb_client,
            embedding_model=self._embedding_model,
            namespace=self.namespace,
        )

    def _init_indexes(self):
        self._dspy_lm = get_dspy_lm_by_llm(self._llm)
        self._kg_index = KnowledgeGraphIndex(
            kg_store=self._kg_store,
            dspy_lm=self._dspy_lm,
            embedding_model=self._embedding_model,
        )

    def class_name(self):
        return "KnowledgeBase"

    def add(
        self,
        source: str | list[str],
        data_type: Optional[DataType] = None,
        loader: Optional[Loader] = None,
        chunker: Optional[Chunker] = None,
    ) -> List[Document]:
        if data_type is None:
            data_type = guess_datatype(source)
        if data_type is None:
            raise ValueError("Please provide a valid data type.")

        if loader is None:
            loader = get_loader_for_datatype(data_type)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            build_index_for_document = partial(
                self.build_index_for_document, chunker=chunker
            )

            results = executor.map(build_index_for_document, loader.load(source))

        return_documents = []
        for result in results:
            return_documents.append(result)
        return return_documents

    def build_index_for_document(
        self,
        document: Document,
        chunker: Optional[Chunker] = None,
    ) -> List[Document]:
        """
        Build index for a document.

        Args:
            document: The document to build index for.
            chunker: The chunker to use to chunk the document.

        Returns:
            A list of documents that are the result of indexing the original document.
        """
        # TODO: handle duplicate documents.
        if chunker is None:
            chunker = get_chunker_for_datatype(document.data_type)

        chunked_document = chunker.chunk(document)
        self.add_document(chunked_document)

        if IndexMethod.KNOWLEDGE_GRAPH in self.index_methods:

            def add_chunk_to_kg(chunk):
                logger.info("Adding chunk <id: %s> to knowledge graph.", chunk.id)
                self._kg_index.add_chunk(chunk)

            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                list(executor.map(add_chunk_to_kg, chunked_document.chunks))

        return chunked_document

    # Document management.

    def add_document(self, document: Document):
        self._doc_store.add([document])

    def add_documents(self, documents: List[Document]):
        return self._doc_store.add(documents)

    def list_documents(self) -> List[Document]:
        return self._doc_store.list()

    def get_document(self, doc_id: uuid.UUID) -> Document:
        return self._doc_store.get(doc_id)

    def delete_document(self, doc_id: uuid.UUID) -> None:
        return self._doc_store.delete(doc_id)

    # Search

    def search(self):
        # TODO: Support one interface search documents and knowledge graph at the same time.
        raise NotImplementedError()

    def search_documents(
        self,
        query: str,
        mode: SearchMode = "vector",
        similarity_threshold: Optional[float] = None,
        num_candidate: Optional[int] = None,
        top_k: Optional[int] = 5,
        **kwargs: Any,
    ) -> DocumentSearchResult:
        return self._doc_store.search(
            query=query,
            mode=mode,
            similarity_threshold=similarity_threshold,
            num_candidate=num_candidate,
            top_k=top_k,
            **kwargs,
        )

    def search_knowledge_graph(
        self,
        query: str,
        depth: int = 2,
        metadata_filters: Optional[dict] = None,
        **kwargs,
    ):
        return self._kg_index.retrieve(
            query=query,
            depth=depth,
            metadata_filters=metadata_filters,
            **kwargs,
        )

    def reset(self):
        self._doc_store.reset()
        self._kg_store.reset()
