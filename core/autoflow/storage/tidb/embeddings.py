from abc import ABC, abstractmethod
from typing import Optional, List

from sqlalchemy import Column
from sqlmodel import Field
from tidb_vector.sqlalchemy import VectorType

from autoflow.llms import EmbeddingModel as _EmbeddingModel


class BaseEmbeddingModel(ABC):
    def SourceField(self, **kwargs):
        return Field(None, **kwargs)

    def VectorField(self, **kwargs):
        return Field(sa_column=Column(VectorType(self._get_dimensions())), **kwargs)

    @abstractmethod
    def _get_dimensions(self) -> int:
        pass

    @abstractmethod
    def get_query_embedding(self, query: str) -> list[float]:
        pass

    @abstractmethod
    def get_source_embedding(self, source: str) -> list[float]:
        pass


class EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, dimensions: Optional[int] = None, **kwargs):
        self.embedding_model = _EmbeddingModel(
            model_name=model_name, dimensions=dimensions, **kwargs
        )
        if dimensions is not None:
            self._dimensions = dimensions
        else:
            self._dimensions = len(self.get_query_embedding("test"))

    def _get_dimensions(self) -> int:
        return self._dimensions

    def get_query_embedding(self, query: str) -> list[float]:
        return self.embedding_model.get_query_embedding(query)

    def get_source_embedding(self, source: str) -> list[float]:
        return self.embedding_model.get_text_embedding(source)

    def get_source_embedding_batch(self, texts: List[str]) -> list[list[float]]:
        return self.embedding_model.get_text_embedding_batch(texts=texts)
