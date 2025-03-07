from abc import ABC, abstractmethod
from typing import List

from autoflow.storage.tidb.base import TiDBModel


class BaseRerankFunction(ABC):
    @abstractmethod
    def rerank(
        self, items: List[TiDBModel], query_str: str, top_n: int = 2
    ) -> List[TiDBModel]:
        pass
