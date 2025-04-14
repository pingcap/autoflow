from pydantic import BaseModel, Field


DEFAULT_CHUNK_SIZE = 1024  # tokens
DEFAULT_CHUNK_OVERLAP = 20  # tokens


class TextChunkerConfig(BaseModel):
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, description="Chunk size")
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP, description="Chunk overlap"
    )
