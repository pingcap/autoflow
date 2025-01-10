"""Jina embeddings file."""

from typing import Any, List, Optional
import requests

from llama_index.core.base.embeddings.base import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.embeddings import BaseEmbedding

MAX_BATCH_SIZE = 500


class _APICaller:
    def __init__(
        self,
        model: str,
        api_url: str,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        self.api_url = api_url
        self.model = model
        self.normalize_embeddings = normalize_embeddings
        self._session = requests.Session()

    def get_embeddings(self, sentences: list[str]) -> List[List[float]]:
        """Get embeddings."""
        # Call Jina AI Embedding API
        resp = self._session.post(  # type: ignore
            self.api_url,
            json={
                "sentences": sentences,
                "model": self.model,
                "normalize_embeddings": self.normalize_embeddings,
            },
        )
        resp.raise_for_status()
        resp_json = resp.json()
        if "embeddings" not in resp_json:
            raise RuntimeError(
                f"Call local embedding api {self.api_url} failed {resp.status_code}"
            )

        return resp_json["embeddings"]

    async def aget_embeddings(self, sentences: list[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        import aiohttp

        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(
                self.api_url,
                json={
                    "sentences": sentences,
                    "model": self.model,
                },
            ) as response:
                resp = await response.json()
                response.raise_for_status()
                return resp["embeddings"]


class LocalEmbedding(BaseEmbedding):
    model: str = Field(
        default="BAAI/bge-m3",
        description="The model to use when calling Jina AI API",
    )

    _encoding_queries: str = PrivateAttr()
    _encoding_documents: str = PrivateAttr()
    _api: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        api_url: str = "http://127.0.0.1:5001/api/v1/embedding",
        normalize_embeddings: bool = True,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model=model,
            **kwargs,
        )
        self._api: _APICaller = _APICaller(
            model=model, api_url=api_url, normalize_embeddings=normalize_embeddings
        )

    @classmethod
    def class_name(cls) -> str:
        return "LocalEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._api.get_embeddings([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        result = await self._api.aget_embeddings([query])
        return result[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        result = await self._aget_text_embeddings([text])
        return result[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._api.get_embeddings(texts)

    async def _aget_text_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        return await self._api.aget_embeddings(texts)
