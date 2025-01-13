from app.models import Chunk
from typing import List
from pydantic import BaseModel


class AppConfig(BaseModel):
    pass


class RetrievalConfig(BaseModel):
    top_k: int = 10
    similarity_top_k: int = None
    metadata_filters: dict = {}
    oversampling_factor: int = 5


class RetrieveRequest(BaseModel):
    query: str
    retrieval_config: RetrievalConfig = RetrievalConfig()


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class RetrieveResponse(BaseModel):
    chunks: List[RetrievedChunk]
