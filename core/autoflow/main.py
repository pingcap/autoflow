import uuid
from typing import Optional, List, TYPE_CHECKING

import sqlalchemy
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel
from autoflow.schema import IndexMethod
from autoflow.storage.tidb.utils import build_tidb_dsn

if TYPE_CHECKING:
    from autoflow.llms.chat_models import ChatModel
    from autoflow.llms.embeddings import EmbeddingModel
    from autoflow.llms import LLMManager


class Autoflow:
    _db_engine = None
    _model_manager = None

    def __init__(
        self, db_engine: Engine, model_manager: Optional["TYPE_CHECKING"] = None
    ):
        from autoflow.llms import default_llm_manager

        self._db_engine = db_engine
        self._model_manager = model_manager or default_llm_manager
        self._init_table_schema()

    @classmethod
    def from_config(
        cls,
        db_host: Optional[str] = None,
        db_port: Optional[int] = None,
        db_username: Optional[str] = None,
        db_password: Optional[str] = None,
        db_name: str = "autoflow",
        db_enable_ssl: Optional[bool] = None,
    ):
        tidb_dsn = str(
            build_tidb_dsn(
                host=db_host,
                port=db_port,
                username=db_username,
                password=db_password,
                database=db_name,
                enable_ssl=db_enable_ssl,
            )
        )
        db_engine = sqlalchemy.create_engine(str(tidb_dsn))
        return cls(
            db_engine=db_engine,
        )

    def _init_table_schema(self):
        SQLModel.metadata.create_all(self._db_engine)

    @property
    def db_engine(self) -> Engine:
        return self._db_engine

    @property
    def llm_manager(self) -> "LLMManager":
        return self._model_manager

    def create_knowledge_base(
        self,
        name: str,
        chat_model: "ChatModel",
        embedding_model: "EmbeddingModel",
        description: Optional[str] = None,
        index_methods: Optional[List[IndexMethod]] = None,
        id: Optional[uuid.UUID] = None,
    ):
        from autoflow.knowledge_base import KnowledgeBase

        return KnowledgeBase(
            name=name,
            description=description,
            index_methods=index_methods,
            chat_model=chat_model,
            embedding_model=embedding_model,
            id=id,
            db_engine=self._db_engine,
        )
