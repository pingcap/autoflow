import enum
from typing import Union, Dict
from pydantic import BaseModel
from sqlmodel import Field
from autoflow.datasources.mime_types import SupportedMimeTypes
from autoflow.llms import EmbeddingModelConfig, ChatModelConfig
from autoflow.schema import IndexMethod

# Index Methods

DEFAULT_INDEX_METHODS = [IndexMethod.VECTOR_SEARCH]

# Chunking Settings

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_HEADER_LEVEL = 2
SENTENCE_CHUNK_OVERLAP = 200
DEFAULT_PARAGRAPH_SEP = "\n\n\n"


class ChunkSplitter(str, enum.Enum):
    SENTENCE_SPLITTER = "SentenceSplitter"
    MARKDOWN_NODE_PARSER = "MarkdownNodeParser"


class SentenceSplitterOptions(BaseModel):
    chunk_size: int = Field(
        description="The token chunk size for each chunk.",
        default=1000,
        gt=0,
    )
    chunk_overlap: int = Field(
        description="The overlap size for each chunk.",
        default=SENTENCE_CHUNK_OVERLAP,
        gt=0,
    )
    paragraph_separator: str = Field(
        description="The paragraph separator for splitting the text.",
        default=DEFAULT_PARAGRAPH_SEP,
    )


class MarkdownNodeParserOptions(BaseModel):
    chunk_size: int = Field(
        description="The token chunk size for each chunk.",
        default=1000,
        gt=0,
    )
    chunk_header_level: int = Field(
        description="The header level to split on",
        default=DEFAULT_CHUNK_HEADER_LEVEL,
        ge=1,
        le=6,
    )


class ChunkSplitterConfig(BaseModel):
    splitter: ChunkSplitter = Field(default=ChunkSplitter.SENTENCE_SPLITTER)
    splitter_options: Union[SentenceSplitterOptions, MarkdownNodeParserOptions] = (
        Field()
    )


class ChunkingMode(str, enum.Enum):
    GENERAL = "general"
    ADVANCED = "advanced"


class BaseChunkingConfig(BaseModel):
    mode: ChunkingMode = Field(default=ChunkingMode.GENERAL)


class GeneralChunkingConfig(BaseChunkingConfig):
    mode: ChunkingMode = Field(default=ChunkingMode.GENERAL)
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0)
    chunk_overlap: int = Field(default=SENTENCE_CHUNK_OVERLAP, gt=0)
    paragraph_separator: str = Field(default=DEFAULT_PARAGRAPH_SEP)


class AdvancedChunkingConfig(BaseChunkingConfig):
    mode: ChunkingMode = Field(default=ChunkingMode.ADVANCED)
    rules: Dict[SupportedMimeTypes, ChunkSplitterConfig] = Field(default_factory=list)


ChunkingConfig = Union[GeneralChunkingConfig | AdvancedChunkingConfig]


# Knowledge Base Config.


class KnowledgeBaseConfig(BaseModel):
    chat_model: ChatModelConfig = None
    embedding_model: EmbeddingModelConfig = None
    chunking_config: ChunkingConfig = GeneralChunkingConfig()
