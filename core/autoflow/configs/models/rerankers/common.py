from pydantic import BaseModel, Field


class BaseRerankerConfig(BaseModel):
    model: str = Field(
        description="The model to use for the reranker",
        default="jina-reranker-v2-base-multilingual",
    )
    top_n: int = Field(
        description="The number of results to return",
        default=5,
    )
