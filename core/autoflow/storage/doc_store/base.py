from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Generic, TypeVar, Type

from pydantic import BaseModel
from autoflow.models.document import Document
from autoflow.storage.schema import QueryBundle

D = TypeVar("D", bound=Type[Document])
C = TypeVar("C")


class DocumentSearchMethod(str, Enum):
    VECTOR_SEARCH = "vector_search"
    FULLTEXT_SEARCH = "fulltext_search"


class DocumentSearchQuery(QueryBundle):
    # document_ids: Optional[List[int]] = None
    # chunk_ids: Optional[List[int]] = None
    # metadata_filters: Optional[MetadataFilters] = None
    search_method: List[DocumentSearchMethod] = [DocumentSearchMethod.VECTOR_SEARCH]
    top_k: Optional[int] = None

    # Vector Search
    similarity_threshold: Optional[float] = None
    # similarity_weight: Optional[float] = None
    similarity_nprobe: Optional[int] = None
    similarity_top_k: Optional[int] = 5

    # Full Text Search
    # TODO: Support Full Text Search.

    # Reranking
    enable_reranker: bool = False
    reranker_model_name: Optional[str] = None


class ChunkWithScore(BaseModel, Generic[C]):
    chunk: C
    score: float


class DocumentSearchResult(BaseModel, Generic[D, C]):
    chunks: List[ChunkWithScore[C]]
    documents: List[Document]


class DocumentStore(ABC, Generic[D, C]):
    @abstractmethod
    def add(self, document: List[D]) -> List[D]:
        raise NotImplementedError()

    @abstractmethod
    def delete(self, document_id: int) -> None:
        raise NotImplementedError()

    @abstractmethod
    def search(self, query: DocumentSearchQuery) -> DocumentSearchResult[D, C]:
        raise NotImplementedError()

    @abstractmethod
    def get(self, document_id: int) -> D:
        raise NotImplementedError()

    @abstractmethod
    def add_doc_chunks(self, chunks: List[C]) -> List[C]:
        raise NotImplementedError()
