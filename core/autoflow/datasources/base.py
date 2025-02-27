from abc import ABC, abstractmethod
from typing import Generator, Generic, TypeVar, Optional
from pydantic import BaseModel

from autoflow.db_models import DBDataSource, DBDocument

C = TypeVar("C", bound=BaseModel)


class DataSource(ABC, Generic[C]):
    _ds: Optional[DBDataSource] = None
    config: C

    def __init__(
        self,
        config: dict,
        ds: Optional[DBDataSource],
    ):
        self._ds = ds
        self.config = self.validate_config(config)

    @classmethod
    def from_db(cls, ds: DBDataSource):
        return cls(ds.config, ds)

    @property
    def id(self):
        return self._ds.id

    @property
    def name(self):
        return self._ds.name

    @abstractmethod
    def validate_config(self, config: dict) -> C:
        raise NotImplementedError()

    @abstractmethod
    def load_documents(self) -> Generator[DBDocument, None, None]:
        raise NotImplementedError
