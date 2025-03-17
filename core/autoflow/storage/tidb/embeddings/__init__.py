from .base import BaseEmbeddingFunction
from .litellm import LiteLLMEmbeddingFunction

EmbeddingFunction = LiteLLMEmbeddingFunction

__all__ = ["BaseEmbeddingFunction", "LiteLLMEmbeddingFunction", "EmbeddingFunction"]
