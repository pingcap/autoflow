from typing import Optional, List
from pydantic import Field
from autoflow.storage.tidb.rerankers.base import BaseRerankFunction
from autoflow.storage.tidb.search import SearchResultModel


class LiteLLMRerankFunction(BaseRerankFunction):
    api_key: str = Field(description="API key for reranker provider")
    api_base: str = Field(description="Base URL for reranker provider")
    timeout: int = Field(description="Timeout for rerank API")

    def __init__(
        self,
        model_name: str,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            dimensions=dimensions,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )
        if dimensions is not None:
            self.dimensions = len(self.get_query_embedding("test"))

    def rerank(
        self, items: List[SearchResultModel], query_str: str, top_n: int = 2, **kwargs
    ) -> List[SearchResultModel]:
        raise NotImplementedError
