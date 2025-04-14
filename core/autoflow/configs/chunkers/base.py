from typing import Optional, Dict

from pydantic import BaseModel, Field, model_validator


class ChunkerConfig(BaseModel):
    provider: str = Field(
        description="Provider of the chunker (e.g., 'text')",
        default="openai",
    )
    config: Optional[Dict] = Field(
        description="Configuration for the specific chunker",
        default=None,
    )

    _provider_configs: Dict[str, str] = {
        "text": "TextChunkerConfig",
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "ChunkerConfig":
        provider = self.provider
        config = self.config

        if provider not in self._provider_configs:
            raise ValueError(f"Unsupported chunker provider: {provider}")

        module = __import__(
            f"autoflow.configs.chunkers.{provider}",
            fromlist=[self._provider_configs[provider]],
        )
        config_class = getattr(module, self._provider_configs[provider])

        if config is None:
            config = {}

        if not isinstance(config, dict):
            if not isinstance(config, config_class):
                raise ValueError(
                    f"Invalid config type for chunker provider: {provider}"
                )
            return self

        self.config = config_class(**config)
        return self
