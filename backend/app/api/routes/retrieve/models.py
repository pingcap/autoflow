from pydantic import BaseModel

from app.rag.indices.knowledge_graph.retriever.schema import (
    KnowledgeGraphRetrieverConfig,
)
from app.rag.indices.vector_search.retriever.schema import VectorSearchRetrieverConfig
from app.rag.knowledge_base.multi_kb_retriever import FusionRetrivalBaseConfig

# Chunks Retrival


class ChunkRetrievalConfig(FusionRetrivalBaseConfig):
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
