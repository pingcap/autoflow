from pydantic import BaseModel

from app.rag.retrievers.chunk.schema import VectorSearchRetrieverConfig


class KBChunkRetrievalConfig(BaseModel):
    vector_search: VectorSearchRetrieverConfig
    # TODO: add fulltext and knowledge graph search config


class KBRetrieveChunksRequest(BaseModel):
    query: str
    retrieval_config: KBChunkRetrievalConfig
