from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

from app.rag.retrievers.knowledge_graph.schema import (
    KnowledgeGraphRetrieverConfig,
)


class SynopsisEntityCreate(BaseModel):
    name: str
    description: str
    topic: str
    meta: dict
    entities: List[int] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_entities(self):
        if len(self.entities) == 0:
            raise ValueError("Entities list should not be empty")
        return self


class EntityUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    meta: Optional[dict] = None


class RelationshipUpdate(BaseModel):
    description: Optional[str] = None
    meta: Optional[dict] = None
    weight: Optional[int] = None


class GraphSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8192)
    include_meta: bool = True
    depth: int = Field(default=2, ge=0, le=3)
    with_degree: bool = True
    relationship_meta_filters: dict = {}


# Knowledge Graph Retrieval


class KBKnowledgeGraphRetrievalConfig(BaseModel):
    knowledge_graph: KnowledgeGraphRetrieverConfig


class KBRetrieveKnowledgeGraphRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8192)
    llm_id: int
    retrieval_config: KBKnowledgeGraphRetrievalConfig


### Experimental


class KnowledgeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8192)
    similarity_threshold: float = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)
    top_k: int = Field(default=10, ge=1, le=100)


class KnowledgeNeighborRequest(BaseModel):
    entities_ids: List[int] = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=8192)
    max_depth: int = Field(default=1, ge=0, le=3)
    max_neighbors: int = Field(default=20, ge=1, le=100)
    similarity_threshold: float = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)


class KnowledgeChunkRequest(BaseModel):
    relationships_ids: List[int] = Field(min_length=1, max_length=100)
