from typing import Dict

from pydantic import BaseModel, Field

from autoflow.configs.models.providers import ProviderConfig


class ManagerConfig(BaseModel):
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
