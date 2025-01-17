from typing import Optional
from pydantic import BaseModel

from app.rag.postprocessors import MetadataFilters


class RerankerConfig(BaseModel):
    model_id: int = None
    top_n: int = 10


class MetadataFilterConfig(BaseModel):
    filters: MetadataFilters = None


class VectorSearchConfig(BaseModel):
    top_k: int = 10
    similarity_top_k: Optional[int] = None
    oversampling_factor: Optional[int] = 5
    reranker: Optional[RerankerConfig] = None
    metadata_filter: Optional[MetadataFilterConfig] = None
