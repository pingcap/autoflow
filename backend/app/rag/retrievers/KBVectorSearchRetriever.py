import logging
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import BaseModel, List

from app.rag.postprocessors.metadata_post_filter import MetadataFilters


logger = logging.getLogger(__name__)


class VectorSearchRerankerConfig(BaseModel):
    enable: bool = True


class VectorSearchMetadataFilterConfig(BaseModel):
    enable: bool = True
    filters: MetadataFilters = None


class VectorSearchRetrieverConfig(BaseModel):
    enable: bool = True
    top_k: int = 10
    similarity_top_k: int = None
    oversampling_factor: int = 5
    reranker: VectorSearchRerankerConfig = None
    metadata_filter: VectorSearchMetadataFilterConfig = None


class KnowledgeGraphRetrieverConfig(BaseModel):
    enable: bool = False


class KnowledgeBaseConfig(BaseModel):
    linked_knowledge_base: LinkedKnowledgeBaseConfig


class RetrieverConfig(BaseModel):
    knowledge_base: KnowledgeBaseConfig
    vector_search: VectorSearchRetrieverConfig
    knowledge_graph: KnowledgeGraphRetrieverConfig


class VectorSearchRetriever(BaseRetriever):
    def __init__(self, config: VectorSearchRetrieverConfig):
        pass


class AppRetriever(BaseRetriever):
    def __init__(
        self,
        config: RetrieverConfig,
    ):
        pass

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        pass
