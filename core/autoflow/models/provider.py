from typing import Dict, Optional

from litellm.types.utils import LlmProviders
from llama_index.core.constants import DEFAULT_TEMPERATURE
from pydantic import BaseModel

from autoflow.models.embeddings import EmbeddingModel
from autoflow.models.llms import LLM
from autoflow.models.rerankers import RerankerModel


ModelProviders = LlmProviders


class ProviderConfig(BaseModel):
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    # TODO: Support Bedrock authenticate.


class LLMConfig(BaseModel):
    provider: str or ModelProviders
    model: str
    max_tokens: Optional[int] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: Optional[int] = 10


class EmbeddingModelConfig(BaseModel):
    provider: str or ModelProviders
    model: str
    dimensions: Optional[int] = None


class RerankerModelConfig(BaseModel):
    provider: str or ModelProviders
    model: str
    top_n: Optional[int] = 5


class ModelManager:
    _registry: Dict[str, ProviderConfig] = {}

    def configure_provider(self, name: ModelProviders, config: ProviderConfig):
        self._registry[name] = config

    def get_provider(self, name: str or ModelProviders) -> Optional[ProviderConfig]:
        provider = self._registry.get(name)
        if provider is None:
            raise ValueError('Provider "{}" is not found.'.format(name))
        return provider

    def resolve_llm(self, config: LLMConfig) -> Optional[LLM]:
        model_provider = self.get_provider(config.provider)
        return LLM(
            model=f"{config.provider}/{config.model}",
            api_base=model_provider.api_base,
            api_key=model_provider.api_key,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            max_retries=config.max_retries,
        )

    def resolve_embedding_model(
        self, config: EmbeddingModelConfig
    ) -> Optional[EmbeddingModel]:
        model_provider = self.get_provider(config.provider)
        return EmbeddingModel(
            model_name=f"{config.provider}/{config.model}",
            api_base=model_provider.api_base,
            api_key=model_provider.api_key,
            dimensions=config.dimensions,
        )

    def resolve_reranker_model(
        self, config: RerankerModelConfig
    ) -> Optional[RerankerModel]:
        model_provider = self.get_provider(config.provider)
        return RerankerModel(
            model=f"{config.provider}/{config.model}",
            top_n=config.top_n,
            api_base=model_provider.api_base,
            api_key=model_provider.api_key,
        )


default_model_manager: ModelManager = ModelManager()
