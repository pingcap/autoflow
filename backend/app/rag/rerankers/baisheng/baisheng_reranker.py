from typing import Any, List, Optional
import requests

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

dispatcher = get_dispatcher(__name__)


class BaishengRerank(BaseNodePostprocessor):
    api_key: str = Field(default="", description="API key.")
    api_url: str = Field(
        default="http://api.chat.prd.yumc.local/chat/v1/reranker",
        description="API url.",
    )
    model: str = Field(
        default="bge-reranker-v2-m3",
        description="The model to use when calling API",
    )

    top_n: int = Field(description="Top N nodes to return.")

    _session: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "bge-reranker-v2-m3",
        api_key: str = "",
        api_url: str = "http://api.chat.prd.yumc.local/chat/v1/reranker",
    ):
        super().__init__(top_n=top_n, model=model)
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    @classmethod
    def class_name(cls) -> str:
        return "BaishengRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model,
            )
        )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            texts = [
                node.node.get_content(metadata_mode=MetadataMode.EMBED)
                for node in nodes
            ]
            resp = self._session.post(  # type: ignore
                self.api_url,
                json={
                    "query": query_bundle.query_str,
                    "model": self.model,
                    "sentences": texts,
                },
            ).json()
            if "scores" not in resp:
                raise RuntimeError(f"Got error from reranker: {resp}")

            results = zip(range(len(nodes)), resp["scores"])
            results = sorted(results, key=lambda x: x[1], reverse=True)[: self.top_n]

            new_nodes = []
            for result in results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result[0]].node, score=result[1]
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
