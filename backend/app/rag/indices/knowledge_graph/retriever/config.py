from typing import Any

from pydantic import BaseModel


class KnowledgeGraphRetrieverConfig(BaseModel):
    depth: int = 2
    include_meta: bool = False
    with_chunks: bool = False
    with_degree: bool = False
    enable_metadata_filter: bool = False
    metadata_filter: Any = None
