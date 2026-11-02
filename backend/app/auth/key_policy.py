from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

ApiScope = Literal["retrieve:chunks", "retrieve:graph"]
PositiveID = Annotated[int, Field(gt=0, strict=True)]


class ApiKeyGrant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scopes: list[ApiScope] = Field(min_length=1, max_length=2)
    knowledge_base_ids: list[PositiveID] = Field(min_length=1, max_length=100)
    expires_in_days: int = Field(default=30, ge=1, le=90, strict=True)
