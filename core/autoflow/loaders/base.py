from abc import abstractmethod
from typing import Generator

from autoflow.types import BaseComponent
from autoflow.storage.doc_store import Document


class Loader(BaseComponent):
    @abstractmethod
    def load(
        self, source: str | list[str], **kwargs
    ) -> Generator[Document, None, None]:
        raise NotImplementedError


class FileLoader(Loader):
    def load(self, files: str | list[str], **kwargs) -> Generator[Document, None, None]:
        if isinstance(files, str):
            files = [files]

        for file in files:
            yield self._load_file(file)

    @abstractmethod
    def _load_file(self, file: str) -> Document:
        raise NotImplementedError
