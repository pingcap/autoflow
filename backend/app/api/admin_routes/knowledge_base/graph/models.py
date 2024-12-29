from typing import List, Optional
from pydantic import BaseModel, model_validator


class SynopsisEntityCreate(BaseModel):
    name: str
    description: str
    topic: str
    meta: dict
    entities: List[int]

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
    query: str
    include_meta: bool = True
    depth: int = 2
    with_degree: bool = True


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


class KnowledgeChunkRequest(BaseModel):
    relationships_ids: List[int]
