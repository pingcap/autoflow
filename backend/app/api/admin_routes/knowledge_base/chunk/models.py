from pydantic import BaseModel
from app.rag.indexes.vector_search.config import VectorSearchConfig


class RetrieveChunkRequest(BaseModel):
    query: str
    vector_search_config: VectorSearchConfig
    # TODO: add fulltext and knowledge graph search config
