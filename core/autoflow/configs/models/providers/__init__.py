from autoflow.configs.models.providers.base import (
    ModelProviders,
    ModelProviderInfo,
    ProviderConfig,
)

model_providers = [
    ModelProviderInfo(
        name=ModelProviders.OPENAI,
        display_name="OpenAI",
        description="The OpenAI API provides a simple interface for developers to create an intelligence layer in their applications, powered by OpenAI's state of the art models.",
        website="https://platform.openai.com",
        supported_model_types=["llm", "text_embedding"],
    )
]

model_provider_mappings = {provider.name: provider for provider in model_providers}

__all__ = [
    "ModelProviders",
    "ModelProviderInfo",
    "ProviderConfig",
    "model_providers",
    "model_provider_mappings",
]
