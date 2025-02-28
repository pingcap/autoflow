import uuid
from abc import abstractmethod
from typing import Generator, Generic, TypeVar, Optional
from pydantic import BaseModel, Field

from autoflow.models import DBDocument
from autoflow.schema import BaseComponent

C = TypeVar("C", bound=BaseModel)


class DataSource(BaseComponent, Generic[C]):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    config: Optional[C] = Field(default=None)

    def __init__(self, config: dict):
        config = self.validate_config(config)
        super().__init__(config=config)

    @abstractmethod
    def validate_config(self, config: dict) -> C:
        raise NotImplementedError()

    @abstractmethod
    def load_documents(self) -> Generator[DBDocument, None, None]:
        raise NotImplementedError
