from typing import Optional, Type

from pydantic import PrivateAttr, Field
from sqlmodel import Session, SQLModel
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import BaseComponent

from app.models import KnowledgeBase as KnowledgeBaseModel
from app.models.chunk import get_dynamic_chunk_model
from app.models.entity import get_dynamic_entity_model
from app.models.relationship import get_dynamic_relationship_model
from app.rag.knowledge.graph.editor import TiDBGraphEditor
from app.rag.storage.graph_stores.tidb_graph_store import TiDBGraphStore
from app.rag.storage.vector_stores.tidb_vector_store import TiDBVectorStore
from app.rag.llms.dspy import get_dspy_lm_by_llama_llm


class KnowledgeBase(BaseComponent):
    id: str = Field()
    name: str = Field()
    vector_dimension: int = Field()

    # Private attributes
    _llm: Optional[LLM] = PrivateAttr()
    _embed_model: Optional[BaseEmbedding] = PrivateAttr()
    _vector_store: Optional[TiDBVectorStore] = PrivateAttr()
    _graph_store: Optional[TiDBGraphStore] = PrivateAttr()
    _chunk_db_model: Optional[Type[SQLModel]] = PrivateAttr()
    _relationship_db_model: Optional[Type[SQLModel]] = PrivateAttr()
    _entity_db_model: Optional[Type[SQLModel]] = PrivateAttr()

    def __init__(
        self,
        id: str,
        name: str,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        db_session: Optional[Session] = None,
    ):
        super().__init__()
        self.id = id
        self.name = name
        self._llm = llm
        self._dspy_lm = get_dspy_lm_by_llama_llm(llm)
        self._embed_model = embed_model
        self.vector_dimension = embed_model.dimension if embed_model else None
        self.namespace = str(id)
        self._chunk_db_model = get_dynamic_chunk_model(
            self.vector_dimension, self.namespace
        )
        self._entity_db_model = get_dynamic_entity_model(
            self.vector_dimension, self.namespace
        )
        self._relationship_db_model = get_dynamic_relationship_model(
            self.vector_dimension, self.namespace, self._entity_db_model
        )
        self._vector_store = TiDBVectorStore(
            session=db_session,
            chunk_db_model=self._chunk_db_model,
        )
        self._graph_store = TiDBGraphStore(
            knowledge_base_id=self.id,
            dspy_lm=self._dspy_lm,
            session=db_session,
            embed_model=self._embed_model,
            entity_db_model=self._entity_db_model,
            relationship_db_model=self._relationship_db_model,
            chunk_db_model=self._chunk_db_model,
        )

    def init(self) -> None:
        self.vector_store.ensure_table_schema()
        self.graph_store.ensure_table_schema()

    @property
    def vector_store(self) -> TiDBVectorStore:
        return self._vector_store

    @property
    def graph_store(self) -> TiDBGraphStore:
        return self._graph_store

    @property
    def graph_editor(self) -> TiDBGraphEditor:
        return TiDBGraphEditor(
            knowledge_base_id=self.id,
            entity_db_model=self.entity_db_model,
            relationship_db_model=self.relationship_db_model,
            embed_model=self.embed_model,
        )

    @property
    def entity_db_model(self) -> Type[SQLModel]:
        return self._entity_db_model

    @property
    def relationship_db_model(self) -> Type[SQLModel]:
        return self._relationship_db_model

    @property
    def chunk_db_model(self) -> Type[SQLModel]:
        return self._chunk_db_model

    @property
    def embed_model(self) -> BaseEmbedding:
        return self._embed_model

    @property
    def dspy_lm(self) -> LLM:
        return self._dspy_lm

    def __repr__(self) -> str:
        return f"KnowledgeBase(id={self.id}, name={self.name})"
