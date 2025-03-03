import uuid
from typing import Optional, List

from sqlalchemy import Engine
from sqlmodel import SQLModel

from autoflow.knowledge_base import KnowledgeBase
from autoflow.llms import (
    ChatModel,
    EmbeddingModel,
)
from autoflow.schema import IndexMethod
from autoflow.llms import LLMManager, default_llm_manager


class Autoflow:
    _db_engine = None
    _model_manager = None

    def __init__(
        self, db_engine: Engine, model_manager: LLMManager = default_llm_manager
    ):
        self._db_engine = db_engine
        self._model_manager = model_manager
        self._init_table_schema()

    def _init_table_schema(self):
        SQLModel.metadata.create_all(self._db_engine)

    @property
    def db_engine(self) -> Engine:
        return self._db_engine

    @property
    def llm_manager(self) -> LLMManager:
        return self._model_manager

    def create_knowledge_base(
        self,
        name: str,
        chat_model: ChatModel,
        embedding_model: EmbeddingModel,
        description: Optional[str] = None,
        index_methods: Optional[List[IndexMethod]] = None,
        id: Optional[uuid.UUID] = None,
    ) -> KnowledgeBase:
        return KnowledgeBase(
            name=name,
            description=description,
            index_methods=index_methods,
            chat_model=chat_model,
            embedding_model=embedding_model,
            id=id,
            db_engine=self._db_engine,
        )
