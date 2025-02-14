import enum
from datetime import datetime
from typing import Annotated, Dict, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import JSON, func
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlmodel import (
    Field,
    Column,
    DateTime,
    Relationship as SQLRelationship,
    SQLModel,
)
from llama_index.core.node_parser.text.sentence import (
    DEFAULT_PARAGRAPH_SEP,
    SENTENCE_CHUNK_OVERLAP,
    CHUNKING_REGEX,
)
from app.rag.node_parser.file.markdown import DEFAULT_CHUNK_HEADER_LEVEL
from app.api.admin_routes.models import KnowledgeBaseDescriptor
from app.exceptions import KBDataSourceNotFound
from app.models.auth import User
from app.models.data_source import DataSource
from app.models.document import ContentFormat
from app.models.embed_model import EmbeddingModel
from app.models.llm import LLM

# For compatibility with old code, define a fake knowledge base id.
PHONY_KNOWLEDGE_BASE_ID = 0


class IndexMethod(str, enum.Enum):
    KNOWLEDGE_GRAPH = "knowledge_graph"
    VECTOR = "vector"


class KnowledgeBaseDataSource(SQLModel, table=True):
    knowledge_base_id: int = Field(primary_key=True, foreign_key="knowledge_bases.id")
    data_source_id: int = Field(primary_key=True, foreign_key="data_sources.id")

    __tablename__ = "knowledge_base_datasources"


class ChunkSplitter(str, enum.Enum):
    SENTENCE_SPLITTER = "sentence-splitter"
    MARKDOWN_SPLITTER = "markdown-splitter"


class BaseSplitterConfig(BaseModel):
    type: ChunkSplitter
    chunk_size: int = Field(
        description="The token chunk size for each chunk.",
        default=1000,
        gt=0,
    )


class SentenceSplitterConfig(BaseSplitterConfig):
    type: Literal[ChunkSplitter.SENTENCE_SPLITTER] = ChunkSplitter.SENTENCE_SPLITTER
    chunk_overlap: int = Field(
        description="The overlap size for each chunk.",
        default=SENTENCE_CHUNK_OVERLAP,
        gt=0,
    )
    separator: str = Field(
        description="The separator for splitting the text.",
        default=" ",
    )
    paragraph_separator: str = Field(
        description="The paragraph separator for splitting the text.",
        default=DEFAULT_PARAGRAPH_SEP,
    )
    secondary_chunking_regex: str = Field(
        description="The regex for secondary chunking.",
        default=CHUNKING_REGEX,
    )


class MarkdownSplitterConfig(BaseSplitterConfig):
    type: Literal[ChunkSplitter.MARKDOWN_SPLITTER] = ChunkSplitter.MARKDOWN_SPLITTER
    chunk_header_level: int = Field(
        description="The header level to split on",
        default=DEFAULT_CHUNK_HEADER_LEVEL,
        ge=1,
        le=6,
    )


ChunkSplitterConfig = Annotated[
    Union[SentenceSplitterConfig, MarkdownSplitterConfig], Field(discriminator="type")
]


default_chunking_rules = {
    ContentFormat.TEXT: SentenceSplitterConfig(),
    ContentFormat.MARKDOWN: MarkdownSplitterConfig(),
}


class ChunkingMode(str, enum.Enum):
    AUTO = "auto"
    ADVANCED = "advanced"


class ChunkingConfig(BaseModel):
    mode: ChunkingMode = Field(default=ChunkingMode.AUTO)


class AutoChunkingConfig(ChunkingConfig):
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=SENTENCE_CHUNK_OVERLAP, gt=0)
    paragraph_separator: str = Field(default=DEFAULT_PARAGRAPH_SEP)


class AdvancedChunkingConfig(ChunkingConfig):
    rules: Dict[ContentFormat, ChunkSplitterConfig] = Field(
        default=default_chunking_rules
    )


class KnowledgeBase(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=255, nullable=False)
    description: str = Field(sa_column=Column(MEDIUMTEXT))

    # The config for transforming the document into (chunk) nodes.
    chunking_config: Dict = Field(
        sa_column=Column(JSON), default=AutoChunkingConfig().model_dump()
    )

    # Data sources config.
    data_sources: list["DataSource"] = SQLRelationship(
        link_model=KnowledgeBaseDataSource
    )

    # Index Config.
    index_methods: list[IndexMethod] = Field(
        default=[IndexMethod.VECTOR], sa_column=Column(JSON)
    )
    llm_id: int = Field(foreign_key="llms.id", nullable=True)
    llm: "LLM" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "foreign_keys": "KnowledgeBase.llm_id",
        },
    )
    embedding_model_id: int = Field(foreign_key="embedding_models.id", nullable=True)
    embedding_model: "EmbeddingModel" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "foreign_keys": "KnowledgeBase.embedding_model_id",
        },
    )
    documents_total: int = Field(default=0)
    data_sources_total: int = Field(default=0)

    # TODO: Support knowledge-base level permission control.

    created_by: UUID = Field(foreign_key="users.id", nullable=True)
    creator: "User" = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "KnowledgeBase.created_by == User.id",
        },
    )
    created_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(), server_default=func.now())
    )
    updated_by: UUID = Field(foreign_key="users.id", nullable=True)
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(), server_default=func.now(), onupdate=func.now()),
    )
    deleted_by: UUID = Field(foreign_key="users.id", nullable=True)
    deleted_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime()))

    __tablename__ = "knowledge_bases"

    def __hash__(self):
        return hash(self.id)

    def get_data_source_by_id(self, data_source_id: int) -> Optional[DataSource]:
        return next(
            (
                ds
                for ds in self.data_sources
                if ds.id == data_source_id and not ds.deleted_at
            ),
            None,
        )

    def must_get_data_source_by_id(self, data_source_id: int) -> DataSource:
        data_source = self.get_data_source_by_id(data_source_id)
        if data_source is None:
            raise KBDataSourceNotFound(self.id, data_source_id)
        return data_source

    def to_descriptor(self) -> KnowledgeBaseDescriptor:
        return KnowledgeBaseDescriptor(
            id=self.id,
            name=self.name,
        )
