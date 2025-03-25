from typing import List, Optional
from pydantic import BaseModel

from app.rag.retrievers.knowledge_graph.schema import (
    KnowledgeGraphRetrieverConfig,
)


class RelationshipUpdate(BaseModel):
    description: Optional[str] = None
    meta: Optional[dict] = None
    weight: Optional[int] = None


class GraphSearchRequest(BaseModel):
    query: str
    include_meta: bool = True
    depth: int = 2
    with_degree: bool = True
    relationship_meta_filters: dict = {}


# Knowledge Graph Retrieval


class KBKnowledgeGraphRetrievalConfig(BaseModel):
    knowledge_graph: KnowledgeGraphRetrieverConfig


class KBRetrieveKnowledgeGraphRequest(BaseModel):
    query: str
    llm_id: int
    retrival_config: KBKnowledgeGraphRetrievalConfig


### Experimental


class KnowledgeRequest(BaseModel):
    query: str
    similarity_threshold: float = 0.55
    top_k: int = 10


class KnowledgeNeighborRequest(BaseModel):
    entities_ids: List[int]
    query: str
    max_depth: int = 1
    max_neighbors: int = 20
    similarity_threshold: float = 0.55


class RelationshipBatchRequest(BaseModel):
    relationship_ids: List[int]
