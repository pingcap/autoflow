"""Jina embeddings file."""

from typing import Any, List, Optional
from app.security.provider_http import ProviderSession
from app.rag.local_model_batches import local_model_batches

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
        api_key: str,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        self.api_url = api_url
        self.model = model
        self.normalize_embeddings = normalize_embeddings
        self._session = ProviderSession(api_url)
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    def get_embeddings(self, sentences: list[str]) -> List[List[float]]:
        """Get embeddings."""
        # Call Jina AI Embedding API
        embeddings = []
        for batch in local_model_batches(sentences):
            with self._session.post(
                self.api_url,
                json={
                    "sentences": batch,
                    "model": self.model,
                    "normalize_embeddings": self.normalize_embeddings,
                },
            ) as response:
                response.raise_for_status()
                values = response.json().get("embeddings")
                if not isinstance(values, list) or len(values) != len(batch):
                    raise RuntimeError(
                        "Local embedding provider returned an invalid response"
                    )
                embeddings.extend(values)
        return embeddings

    async def aget_embeddings(self, sentences: list[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        import asyncio

        return await asyncio.to_thread(self.get_embeddings, sentences)


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
        api_key: str = "",
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
            model=model,
            api_url=api_url,
            api_key=api_key,
            normalize_embeddings=normalize_embeddings,
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
