from typing import List, Optional
from sqlalchemy.engine import Engine

from autoflow.configs.db import DatabaseConfig
from autoflow.configs.knowledge_base import IndexMethod
from autoflow.configs.main import Config
from autoflow.db import get_db_engine_from_config
from autoflow.knowledge_base.base import KnowledgeBase
from autoflow.models.embedding_models import EmbeddingModel
from autoflow.models.llms import LLM
from autoflow.models.manager import ModelManager, model_manager as default_model_manager
from autoflow.models.rerank_models import RerankModel


class Autoflow:
    _db_engine = None

    def __init__(
        self,
        db_engine: Engine,
        model_manager: Optional[ModelManager] = None,
    ):
        self._db_engine = db_engine
        self._model_manager = model_manager or default_model_manager

    @classmethod
    def from_config(cls, config: Config) -> "Autoflow":
        db_engine = cls._init_db_engine(config.db)
        model_manager = ModelManager()
        return cls(db_engine=db_engine, model_manager=model_manager)

    @classmethod
    def _init_db_engine(cls, db_config: DatabaseConfig) -> Engine:
        if db_config.provider != "tidb":
            raise NotImplementedError(
                f"Unsupported database provider: {db_config.provider}."
            )
        return get_db_engine_from_config(db_config)

    @property
    def db_engine(self) -> Engine:
        return self._db_engine

    @property
    def llm_manager(self) -> "ModelManager":
        return self._model_manager

    def create_knowledge_base(
        self,
        name: str,
        namespace: Optional[str] = None,
        description: Optional[str] = None,
        index_methods: Optional[List[IndexMethod]] = None,
        llm: Optional[LLM] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        rerank_model: Optional[RerankModel] = None,
    ):
        return KnowledgeBase(
            db_engine=self.db_engine,
            namespace=namespace,
            name=name,
            description=description,
            index_methods=index_methods,
            llm=llm,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
        )
