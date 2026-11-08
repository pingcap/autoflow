from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

ApiScope = Literal["retrieve:chunks", "retrieve:graph", "chat:write"]
PositiveID = Annotated[int, Field(gt=0, strict=True)]


class ApiKeyGrant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scopes: list[ApiScope] = Field(min_length=1, max_length=3)
    knowledge_base_ids: list[PositiveID] = Field(default_factory=list, max_length=100)
    chat_engine_ids: list[PositiveID] = Field(default_factory=list, max_length=100)
    expires_in_days: int = Field(default=30, ge=1, le=90, strict=True)

    @model_validator(mode="after")
    def require_resource_scope(self):
        if (
            any(scope.startswith("retrieve:") for scope in self.scopes)
            and not self.knowledge_base_ids
        ):
            raise ValueError("Retrieval keys require explicit knowledge bases")
        if "chat:write" in self.scopes and not self.chat_engine_ids:
            raise ValueError("Chat keys require explicit chat engines")
        return self
