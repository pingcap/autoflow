from abc import ABC, abstractmethod
from uuid import UUID
from typing import Generator, Any
from sqlmodel import Session

from app.models import Document


class BaseDataSource(ABC):
    session: Session
    knowledge_base_id: int
    data_source_id: int
    user_id: UUID
    config: Any

    def __init__(
        self,
        session: Session,
        knowledge_base_id: int,
        data_source_id: int,
        user_id: UUID,
        config: Any,
        **kwargs,
    ):
        self.config = config
        self.session = session
        self.knowledge_base_id = knowledge_base_id
        self.data_source_id = data_source_id
        self.user_id = user_id
        self.validate_config()

    @abstractmethod
    def validate_config(self):
        raise NotImplementedError

    @abstractmethod
    def load_documents(self) -> Generator[Document, None, None]:
        raise NotImplementedError
