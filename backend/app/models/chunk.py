import enum
from functools import lru_cache

from typing import Optional, Type
from sqlmodel import (
    Field,
    Column,
    Text,
    JSON,
    Relationship as SQLRelationship,
    SQLModel,
)
from tidb_vector.sqlalchemy import VectorType
from llama_index.core.schema import TextNode

from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_scoped.table_naming import get_kb_vector_dims
from app.utils.namespace import format_namespace
from .base import UpdatableBaseModel, UUIDBaseModel
from app.logger import logger


class KgIndexStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def get_kb_chunk_model(kb: KnowledgeBase) -> Type[SQLModel]:
    vector_dimension = get_kb_vector_dims(kb)
    return get_dynamic_chunk_model(vector_dimension, str(kb.id))


@lru_cache(maxsize=None)
def get_dynamic_chunk_model(
    vector_dimension: int,
    namespace: Optional[str] = None,
) -> Type[SQLModel]:
    namespace = format_namespace(namespace)
    chunk_table_name = f"chunks_{namespace}"
    chunk_model_name = f"Chunk_{namespace}_{vector_dimension}"

    logger.info(
        "Dynamic create chunk model (dimension: %s, table: %s, model: %s)",
        vector_dimension,
        chunk_table_name,
        chunk_model_name,
    )

    class Chunk(UUIDBaseModel, UpdatableBaseModel):
        hash: str = Field(max_length=64)
        text: str = Field(sa_column=Column(Text))
        meta: dict = Field(default={}, sa_column=Column(JSON))
        embedding: list[float] = Field(sa_type=VectorType(vector_dimension))
        document_id: int = Field(foreign_key="documents.id", nullable=True)
        relations: dict | list = Field(default={}, sa_column=Column(JSON))
        source_uri: str = Field(max_length=512, nullable=True)

        # TODO: Add vector_index_status, vector_index_result column, vector index should be optional in the future.

        # TODO: Rename to kg_index_status, kg_index_result column.
        index_status: KgIndexStatus = KgIndexStatus.NOT_STARTED
        index_result: str = Field(sa_column=Column(Text, nullable=True))

        def to_llama_text_node(self) -> TextNode:
            return TextNode(
                id_=self.id.hex,
                text=self.text,
                embedding=list(self.embedding),
                metadata=self.meta,
            )

    chunk_model = type(
        chunk_model_name,
        (Chunk,),
        {
            "__tablename__": chunk_table_name,
            "__table_args__": {"extend_existing": True},
            "__annotations__": {
                "document": Document,
            },
            "document": SQLRelationship(
                sa_relationship_kwargs={
                    "lazy": "joined",
                    "primaryjoin": f"{chunk_model_name}.document_id == Document.id",
                },
            ),
        },
        table=True,
    )

    return chunk_model
