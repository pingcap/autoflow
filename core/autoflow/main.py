from typing import Optional, List

from sqlalchemy import Engine
from sqlmodel import SQLModel, Session

from autoflow.knowledge_base import KnowledgeBase
from autoflow.knowledge_base.config import IndexMethod, EmbeddingModelConfig, LLMConfig
from autoflow.db_models import DBKnowledgeBase
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
    ) -> KnowledgeBase:
        return KnowledgeBase(
            name=name,
            description=description,
            index_methods=index_methods,
            llm=llm,
            embedding_model=embedding_model,
            db_engine=self._db_engine,
        )

    def get_knowledge_base(self, kb_id: int) -> KnowledgeBase:
        with Session(self._db_engine, expire_on_commit=False) as db_session:
            kb = db_session.get(DBKnowledgeBase, kb_id)
            return KnowledgeBase(
                name=kb.name,
                description=kb.description,
                index_methods=kb.index_methods,
                llm=kb.llm,
                embedding_model=kb.embedding_model,
                db_engine=kb.db_engine,
                model_manager=kb.model_manager,
                kb=kb,
            )
