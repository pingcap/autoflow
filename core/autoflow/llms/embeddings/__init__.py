from typing import Optional, List

from llama_index.embeddings.litellm import LiteLLMEmbedding

from autoflow.storage.tidb.embed_fn import BaseEmbeddingFunction

EmbeddingModel = LiteLLMEmbedding


class EmbeddingFunction(BaseEmbeddingFunction):
    def __init__(self, model_name: str, dimensions: Optional[int] = None, **kwargs):
        self.embedding_model = LiteLLMEmbedding(
            model_name=model_name, dimensions=dimensions, **kwargs
        )
        if dimensions is not None:
            self._dimensions = dimensions
        else:
            self._dimensions = len(self.get_query_embedding("test"))

    def _get_dimensions(self) -> int:
        return self._dimensions

    def get_query_embedding(self, query: str) -> list[float]:
        return self.embedding_model.get_query_embedding(query)

    def get_source_embedding(self, source: str) -> list[float]:
        return self.embedding_model.get_text_embedding(source)

    def get_source_embedding_batch(self, sources: List[str]) -> list[list[float]]:
        return self.embedding_model.get_text_embedding_batch(texts=sources)


__all__ = ["EmbeddingModel", "EmbeddingFunction"]
