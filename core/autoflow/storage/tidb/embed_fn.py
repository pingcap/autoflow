from abc import ABC, abstractmethod
from typing import Optional

from sqlalchemy import Column
from sqlmodel import Field
from tidb_vector.sqlalchemy import VectorType


class BaseEmbeddingFunction(ABC):
    def VectorField(self, source_field: Optional[str] = None, **kwargs):
        dimensions = self._get_dimensions()
        return Field(
            sa_column=Column(VectorType(self._get_dimensions())),
            schema_extra={
                "embed_fn": self,
                "dimensions": dimensions,
                "source_field": source_field,
            },
            **kwargs,
        )

    @abstractmethod
    def _get_dimensions(self) -> int:
        pass

    @abstractmethod
    def get_query_embedding(self, query: str) -> list[float]:
        pass

    @abstractmethod
    def get_source_embedding(self, source: str) -> list[float]:
        pass

    @abstractmethod
    def get_source_embedding_batch(self, source: str) -> list[float]:
        pass
