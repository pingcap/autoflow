from typing import Optional, Literal, List

from pydantic import BaseModel, Field
from litellm import LlmProviders

ModelProviders = LlmProviders

ModelType = Literal["llm", "text_embedding", "rerank"]


class ModelProviderInfo(BaseModel):
    name: ModelProviders = Field(
        description="The name of the model provider.",
    )
    logo: Optional[str] = Field(
        description="The logo of the model provider", default=None
    )
    display_name: str = Field(
        description="The name of the model provider",
    )
    description: str = Field(
        description="The description of the model provider", default=None
    )
    website: Optional[str] = Field(
        description="The website of the model provider", default=None
    )
    supported_model_types: List[ModelType] = Field(
        description="The model types supported by the model provider"
    )


class ProviderConfig(BaseModel):
    api_key: Optional[str] = Field(
        title="API key",
        default=None,
    )
    api_base: Optional[str] = Field(
        title="API Base",
        default=None,
    )
