from typing import Optional
from pydantic import BaseModel


class RerankerConfig(BaseModel):
    model_id: int = None
    top_n: int = 10


class MetadataFilterConfig(BaseModel):
    filters: dict = None


class VectorSearchRetrieverConfig(BaseModel):
    top_k: int = 10
    similarity_top_k: Optional[int] = None
    oversampling_factor: Optional[int] = 5
    reranker: Optional[RerankerConfig] = None
    metadata_filter: Optional[MetadataFilterConfig] = None


class RetrievedChunkDocument(BaseModel):
    id: int
    name: str
    source_uri: str


class RetrievedChunk(BaseModel):
    id: str
    text: str
    metadata: dict
    document: RetrievedChunkDocument
    score: float
