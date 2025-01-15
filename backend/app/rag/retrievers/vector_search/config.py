from pydantic import BaseModel

from app.rag.postprocessors import MetadataFilters


class RerankerConfig(BaseModel):
    reranker_model_id: int = None
    top_n: int = 10


class MetadataFilterConfig(BaseModel):
    filters: MetadataFilters = None


class VectorSearchConfig(BaseModel):
    knowledge_base_id: int
    top_k: int = 10
    similarity_top_k: int = None
    oversampling_factor: int = 5
    reranker: RerankerConfig = None
    metadata_filter: MetadataFilterConfig = None
