import uuid
from typing import Optional, List

from sqlalchemy import Engine
from sqlmodel import SQLModel

from autoflow.knowledge_base import KnowledgeBase
from autoflow.knowledge_base.config import EmbeddingModelConfig, LLMConfig
from autoflow.schema import IndexMethod
from autoflow.models import ModelManager, default_model_manager


class Autoflow:
    _db_engine = None
    _model_manager = None

    def __init__(
        self, db_engine: Engine, model_manager: ModelManager = default_model_manager
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
    def model_manager(self) -> ModelManager:
        return self._model_manager

    def crate_knowledge_base(
        self,
        name: str,
        description: Optional[str] = None,
        index_methods: Optional[List[IndexMethod]] = None,
        llm: LLMConfig = None,
        embedding_model: EmbeddingModelConfig = None,
        kb_id: Optional[uuid.UUID] = None,
    ) -> KnowledgeBase:
        return KnowledgeBase(
            name=name,
            description=description,
            index_methods=index_methods,
            llm=llm,
            embedding_model=embedding_model,
            kb_id=kb_id,
            db_engine=self._db_engine,
        )
