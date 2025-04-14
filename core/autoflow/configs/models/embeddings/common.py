from typing import Optional

from pydantic import BaseModel, Field


class BaseEmbeddingConfig(BaseModel):
    model: str = Field(
        description="The model to use for the embedding",
        default="text-embedding-3-small",
    )
    max_tokens: Optional[int] = None
    temperature: float = 0.1
