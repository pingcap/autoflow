from typing import Optional, Dict

from pydantic import BaseModel, Field, model_validator
from autoflow.configs.models.providers import ModelProviders


class RerankerConfig(BaseModel):
    provider: ModelProviders = Field(
        description="Provider of the reranker models (e.g., 'openai')",
        default=ModelProviders.OPENAI,
    )
    config: Optional[Dict] = Field(
        description="Configuration for the specific reranker model",
        default=None,
    )
    _provider_configs: Dict[str, str] = {
        "jina_ai": "JinaAIRerankerConfig",
        "custom": "CustomRerankerConfig",
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "RerankerConfig":
        provider = self.provider.value
        config = self.config

        if provider not in self._provider_configs:
            raise ValueError(f"Unsupported reranker provider: {provider}")

        module = __import__(
            f"autoflow.configs.models.rerankers.{provider}",
            fromlist=[self._provider_configs[provider]],
        )
        config_class = getattr(module, self._provider_configs[provider])

        if config is None:
            config = {}

        if not isinstance(config, dict):
            if not isinstance(config, config_class):
                raise ValueError(
                    f"Invalid config type for reranker provider: {provider}"
                )
            return self

        self.config = config_class(**config)
        return self
