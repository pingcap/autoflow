from typing import Optional, Dict

from pydantic import BaseModel, Field, model_validator

from autoflow.configs.models.providers import ModelProviders

DEFAULT_TEMPERATURE = 0.1


class LLMConfig(BaseModel):
    provider: ModelProviders = Field(
        description="Provider of the large language models (LLM) (e.g., 'openai')",
        default=ModelProviders.OPENAI,
    )
    config: Optional[Dict] = Field(
        description="Configuration for the specific database",
        default=None,
    )
    _llm_configs: Dict[str, str] = {
        "openai": "OpenAILLMConfig",
        "custom": "CustomLLMConfig",
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "LLMConfig":
        provider = self.provider.value
        config = self.config

        if provider not in self._llm_configs:
            raise ValueError(f"Unsupported llm provider: {provider}")

        module = __import__(
            f"autoflow.configs.models.llms.{provider}",
            fromlist=[self._llm_configs[provider]],
        )
        config_class = getattr(module, self._llm_configs[provider])

        if config is None:
            config = {}

        if not isinstance(config, dict):
            if not isinstance(config, config_class):
                raise ValueError(f"Invalid config type for llm provider: {provider}")
            return self

        self.config = config_class(**config)
        return self
