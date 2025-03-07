from typing import List

from .litellm import LiteLLMReranker
from ...storage.tidb.base import TiDBModel
from ...storage.tidb.rerank_fn import BaseRerankFunction

RerankerModel = LiteLLMReranker


class RerankFunction(BaseRerankFunction):
    def __init__(self, model: str):
        self._model = model

    def rerank(
        self, items: List[TiDBModel], query_str: str, top_n: int = 2
    ) -> List[TiDBModel]:
        raise NotImplementedError("Rerank function is not implemented.")
        # reranker_model = RerankerModel(model=self._model, top_n=top_n)
        # nodes = [
        #     NodeWithScore(
        #         node=TextNode(
        #             text=item.sou
        #         )
        #     )
        #     for item in items
        # ]
        # reranker_model.postprocess_nodes(nodes=)
        # return items


__all__ = ["RerankerModel"]
