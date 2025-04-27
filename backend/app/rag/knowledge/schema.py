from enum import Enum


class KBIndexType(str, Enum):
    VECTOR_SEARCH = "VECTOR_SEARCH"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
