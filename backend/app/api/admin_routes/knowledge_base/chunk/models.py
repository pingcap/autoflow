from pydantic import BaseModel, Field

from app.rag.retrievers.chunk.schema import VectorSearchRetrieverConfig


class KBChunkRetrievalConfig(BaseModel):
    vector_search: VectorSearchRetrieverConfig
    score_threshold: float = Field(gt=0, lt=1, default=0.3)
    # TODO: add fulltext and knowledge graph search config


class KBRetrieveChunksRequest(BaseModel):
    query: str
    retrieval_config: KBChunkRetrievalConfig
