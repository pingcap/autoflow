from abc import abstractmethod

from autoflow.types import BaseComponent
from autoflow.storage.doc_store import Document


class Chunker(BaseComponent):
    @abstractmethod
    def chunk(self, document: Document) -> Document:
        raise NotImplementedError
