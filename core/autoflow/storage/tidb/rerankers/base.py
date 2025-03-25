from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from autoflow.storage.tidb.search import SearchResultModel


class BaseRerankFunction(BaseModel, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def rerank(
        self, items: List[SearchResultModel], query_str: str, top_n: int = 2
    ) -> List[SearchResultModel]:
        pass
