from typing import Optional

from pydantic import Field, BaseModel


class BaseLLMConfig(BaseModel):
    model: str = Field(
        description="The model to use for the LLM",
        default="gpt-4.1-2025-04-14",
    )
    max_tokens: Optional[int] = None
    temperature: float = 0.1
