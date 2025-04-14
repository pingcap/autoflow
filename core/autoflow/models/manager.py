from typing import Dict, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from autoflow.configs.models.providers import ModelProviders
from autoflow.configs.models.embeddings import EmbeddingModelConfig
from autoflow.configs.models.llms import LLMConfig
from autoflow.configs.models.providers.base import ProviderConfig
from autoflow.configs.models.rerankers import RerankerConfig

from autoflow.models.embedding_models import EmbeddingModel
from autoflow.models.llms import LLM
from autoflow.models.rerank_models import RerankModel


class ModelManager:
    _registry: Dict[ModelProviders, ProviderConfig] = {}

    @classmethod
    def load_from_db(cls):
        pass

    @classmethod
    def from_config(cls):
        pass

    def registry_provider(self, name: ModelProviders, config: ProviderConfig):
        self._registry[name] = config

    def get_provider_config(self, name: ModelProviders) -> Optional[ProviderConfig]:
        provider = self._registry.get(name)
        if provider is None:
            raise ValueError('Provider "{}" is not found.'.format(name))
        return provider

    def resolve_llm(
        self,
        provider: Optional[ModelProviders] = ModelProviders.OPENAI,
        config: Optional[Dict] = None,
    ) -> Optional[BaseLLM]:
        cfg = LLMConfig.model_validate(
            {
                "provider": provider,
                "config": config,
            }
        )
        provider_config = self.get_provider_config(cfg.provider)
        merged_config = {
            **provider_config.model_dump(),
            **cfg.config.model_dump(),
            "model": f"{cfg.provider.value}/{cfg.config.model}",
        }
        return LLM(**merged_config)

    def resolve_embedding_model(
        self,
        provider: Optional[ModelProviders] = ModelProviders.OPENAI,
        config: Optional[Dict] = None,
    ) -> Optional[BaseEmbedding]:
        cfg = EmbeddingModelConfig.model_validate(
            {
                "provider": provider,
                "config": config,
            }
        )

        provider_config = self.get_provider_config(cfg.provider)
        merged_config = {
            **provider_config.model_dump(),
            **cfg.config.model_dump(exclude={"model"}),
            "model_name": f"{cfg.provider.value}/{cfg.config.model}",
        }
        return EmbeddingModel(**merged_config)

    def resolve_rerank_model(
        self,
        provider: Optional[ModelProviders] = ModelProviders.OPENAI,
        config: Optional[Dict] = None,
    ) -> Optional[BaseNodePostprocessor]:
        cfg = RerankerConfig.model_validate(
            {
                "provider": provider,
                "config": config,
            }
        )
        provider_config = self.get_provider_config(cfg.provider)
        merged_config = {
            **provider_config.model_dump(),
            **cfg.config.model_dump(exclude={"model"}),
            "model": f"{cfg.provider.value}/{cfg.config.model}",
        }
        return RerankModel(**merged_config)


model_manager: ModelManager = ModelManager()
