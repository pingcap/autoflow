from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI, AsyncOpenAI


class OpenAILikeEmbedding(BaseEmbedding):
    # We cannot directly call the llama-index's API because it limited the model name
    # And the 'embedding-2' or 'embedding-3' is not one of the OpenAI's model name

    model: str = Field(
        default="embedding-3",
        description="The model to use when calling Zhipu AI API",
    )
    _client: OpenAI = PrivateAttr()
    _aclient: AsyncOpenAI = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        model: str = "embedding-3",
        api_base: str = "https://open.bigmodel.cn/api/paas/v4/",
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

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._aclient = AsyncOpenAI(api_key=api_key, base_url=api_base)

    def get_embeddings(self, sentences: list[str]) -> List[List[float]]:
        """Get embeddings."""
        # Call Zhipu AI Embedding API via OpenAI client
        embedding_objs = self._client.embeddings.create(
            input=sentences, model=self.model
        ).data
        embeddings = [obj.embedding for obj in embedding_objs]

        return embeddings

    async def aget_embeddings(self, sentences: list[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        result = await self._aclient.embeddings.create(
            input=sentences, model=self.model
        )
        embeddings = [obj.embedding for obj in result.data]

        return embeddings

    @classmethod
    def class_name(cls) -> str:
        return "OpenAILikeEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.get_embeddings([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        result = await self.aget_embeddings([query])
        return result[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        result = await self._aget_text_embeddings([text])
        return result[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.get_embeddings(texts)

    async def _aget_text_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        return await self.aget_embeddings(texts)
