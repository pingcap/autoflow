from typing import Optional
from sqlalchemy.engine import Engine

from autoflow.configs.db import DatabaseConfig
from autoflow.configs.main import Config
from autoflow.db import get_db_engine_from_config
from autoflow.models.manager import ModelManager, model_manager as default_model_manager


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
        model_manager = ModelManager.from_config({})
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
    ):
        pass
