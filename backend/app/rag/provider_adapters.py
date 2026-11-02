"""Bounded transports for provider integrations that do not expose client options."""

import asyncio
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.embeddings.jinaai.base import _JinaAPICaller
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.xinference_rerank import XinferenceRerank
from llama_index.core.schema import MetadataMode, NodeWithScore
from app.security.provider_http import ProviderSession, openai_http_clients


class BoundedJinaCaller(_JinaAPICaller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        previous = self._session
        self._session = ProviderSession(self.api_url)
        self._session.headers.update(previous.headers)
        previous.close()

    async def aget_embeddings(self, *args, **kwargs):
        # Use the same bounded, non-redirecting implementation in both modes.
        return await asyncio.to_thread(self.get_embeddings, *args, **kwargs)


class BoundedJinaEmbedding(JinaEmbedding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        previous = self._api
        self._api = BoundedJinaCaller(
            model=previous.model, api_key=previous.api_key, base_url=kwargs["base_url"]
        )
        previous._session.close()


class BoundedJinaRerank(JinaRerank):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        previous = self._session
        self._session = ProviderSession(self.api_url)
        self._session.headers.update(previous.headers)
        previous.close()


class BoundedCohereRerank(CohereRerank):
    def __init__(self, **kwargs):
        from cohere import Client
        import httpx

        super().__init__(**kwargs)
        self._client = Client(
            api_key=kwargs["api_key"],
            base_url=kwargs["base_url"],
            timeout=60,
            httpx_client=httpx.Client(
                timeout=60, trust_env=False, follow_redirects=False
            ),
        )


class BoundedXinferenceRerank(XinferenceRerank):
    def _postprocess_nodes(self, nodes, query_bundle=None):
        if not nodes:
            return []
        with ProviderSession(self.base_url) as session:
            response = session.post(
                self.base_url.rstrip("/") + "/v1/rerank",
                json={
                    "model": self.model,
                    "query": self.get_query_str(query_bundle),
                    "documents": [
                        node.node.get_content(metadata_mode=MetadataMode.EMBED)
                        for node in nodes
                    ],
                },
            )
            response.raise_for_status()
            results = response.json()["results"][: self.top_n]
        if any(
            type(item["index"]) is not int or not 0 <= item["index"] < len(nodes)
            for item in results
        ):
            raise ValueError("Invalid reranker result")
        return [
            NodeWithScore(node=nodes[item["index"]].node, score=item["relevance_score"])
            for item in results
        ]


def cohere_http_clients():
    clients = openai_http_clients()
    return {
        "httpx_client": clients["http_client"],
        "httpx_async_client": clients["async_http_client"],
        "timeout": 60,
    }
