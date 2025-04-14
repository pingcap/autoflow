from uuid import UUID
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from autoflow.storage.doc_store.types import Chunk, Document, DocumentSearchResult


class DocumentStore(ABC):
    @abstractmethod
    def add(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError()

    @abstractmethod
    def update(self, document_id: UUID, update: Dict[str, Any]):
        raise NotImplementedError()

    @abstractmethod
    def delete(self, document_id: UUID) -> None:
        raise NotImplementedError()

    @abstractmethod
    def list(self, filters: Dict[str, Any] = None) -> List[Document]:
        raise NotImplementedError()

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_candidate: Optional[int] = None,
    ) -> DocumentSearchResult:
        raise NotImplementedError()

    @abstractmethod
    def get(self, document_id: UUID) -> Document:
        raise NotImplementedError()

    @abstractmethod
    def add_doc_chunks(self, document_id: UUID, chunks: List[Chunk]) -> List[Chunk]:
        raise NotImplementedError()

    @abstractmethod
    def list_doc_chunks(self, document_id: UUID) -> List[Chunk]:
        raise NotImplementedError()

    @abstractmethod
    def get_chunk(self, chunk_id: UUID) -> Chunk:
        raise NotImplementedError()

    @abstractmethod
    def update_chunk(self, chunk_id: UUID, update: Dict[str, Any]) -> Chunk:
        raise NotImplementedError()

    @abstractmethod
    def delete_chunk(self, chunk_id: UUID) -> None:
        raise NotImplementedError()
