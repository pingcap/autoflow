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


class VLLMRerank(BaseNodePostprocessor):
    base_url: str = Field(default="", description="The base URL of vLLM API.")
    model: str = Field(default="", description="The model to use when calling API.")

    top_n: int = Field(description="Top N nodes to return.")

    _session: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = "http://localhost:8000",
    ):
        super().__init__(top_n=top_n, model=model)
        self.base_url = base_url
        self.model = model
        self._session = requests.Session()

    @classmethod
    def class_name(cls) -> str:
        return "VLLMRerank"

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
                url=f"{self.base_url}/v1/score",
                json={
                    "text_1": query_bundle.query_str,
                    "model": self.model,
                    "text_2": texts,
                },
            )
            resp.raise_for_status()
            resp_json = resp.json()
            if "data" not in resp_json:
                raise RuntimeError(f"Got error from reranker: {resp_json}")

            results = zip(range(len(nodes)), resp_json["data"])
            results = sorted(results, key=lambda x: x[1]["score"], reverse=True)[
                : self.top_n
            ]

            new_nodes = []
            for result in results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result[0]].node, score=result[1]["score"]
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
