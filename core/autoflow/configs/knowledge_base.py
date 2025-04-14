from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from autoflow.configs.models.llms import LLMConfig
from autoflow.configs.models.embeddings import EmbeddingModelConfig
from autoflow.configs.models.rerankers import RerankerConfig

# Index Methods


class IndexMethod(str, Enum):
    VECTOR_SEARCH = "VECTOR_SEARCH"
    FULLTEXT_SEARCH = "FULLTEXT_SEARCH"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"


DEFAULT_INDEX_METHODS = [IndexMethod.VECTOR_SEARCH]

# Knowledge Base Config


class Version(int, Enum):
    V1 = 1


class KnowledgeBaseConfig(BaseModel):
    version: int = Version.V1
    name: str
    description: Optional[str] = None
    index_methods: List[IndexMethod] = DEFAULT_INDEX_METHODS
    llm: LLMConfig = None
    embedding_model: EmbeddingModelConfig = None
    reranker: RerankerConfig = None
