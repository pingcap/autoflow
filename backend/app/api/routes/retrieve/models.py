from typing import Optional

from pydantic import BaseModel

from app.rag.retrievers.knowledge_graph.schema import (
    KnowledgeGraphRetrieverConfig,
)
from app.rag.retrievers.chunk.schema import VectorSearchRetrieverConfig
from app.rag.retrievers.multiple_knowledge_base import FusionRetrievalBaseConfig

# Chunks retrieval


class ChunkRetrievalConfig(FusionRetrievalBaseConfig):
    full_documents: Optional[bool] = False
    vector_search: VectorSearchRetrieverConfig


class ChunksRetrievalRequest(BaseModel):
    query: str
    retrieval_config: ChunkRetrievalConfig


## Knowledge Graph retrieval


class KnowledgeGraphRetrievalConfig(FusionRetrievalBaseConfig):
    knowledge_graph: KnowledgeGraphRetrieverConfig


class KnowledgeGraphRetrievalRequest(BaseModel):
    query: str
    retrieval_config: KnowledgeGraphRetrievalConfig
