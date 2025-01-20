from typing import List
from app.api.admin_routes.knowledge_base.chunk.models import KBRetrieveChunksRequest
from app.api.admin_routes.knowledge_base.graph.models import (
    KBKnowledgeGraphRetrievalConfig,
)


class RetrieveChunksRequest(KBRetrieveChunksRequest):
    knowledge_base_ids: List[int]


class RetrieveKnowledgeGraphRequest(KBKnowledgeGraphRetrievalConfig):
    knowledge_base_ids: List[int]
