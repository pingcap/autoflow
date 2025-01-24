from typing import Optional

from pydantic import BaseModel

from app.rag.retrievers.knowledge_graph.schema import (
    KnowledgeGraphRetrieverConfig,
)
from app.rag.retrievers.chunk.schema import VectorSearchRetrieverConfig
from app.rag.retrievers.multiple_knowledge_base import FusionRetrivalBaseConfig

# Chunks Retrival


class ChunkRetrievalConfig(FusionRetrivalBaseConfig):
    full_documents: Optional[bool] = False
    vector_search: VectorSearchRetrieverConfig


class ChunksRetrivalRequest(BaseModel):
    query: str
    retrieval_config: ChunkRetrievalConfig


## Knowledge Graph Retrival


class KnowledgeGraphRetrievalConfig(FusionRetrivalBaseConfig):
    knowledge_graph: KnowledgeGraphRetrieverConfig


class KnowledgeGraphRetrivalRequest(BaseModel):
    query: str
    retrieval_config: KnowledgeGraphRetrievalConfig
