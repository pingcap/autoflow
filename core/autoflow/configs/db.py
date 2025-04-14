from typing import Optional

from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    database_url: Optional[str] = Field(
        description="Database connection string",
        default=None,
    )
    provider: Optional[str] = Field(
        description="Database provider",
        default="tidb",
    )
    host: Optional[str] = Field(
        description="Database host.",
        default="localhost",
    )
    port: Optional[int] = Field(description="Database port.", default=4000)
    username: Optional[str] = Field(
        description="The username to connect the database.",
        default="root",
    )
    password: Optional[str] = Field(
        description="The password to connect the database.",
        default="",
    )
    database: str = Field(
        description="Default name for the database",
        default="autoflow",
    )
    enable_ssl: Optional[bool] = Field(
        description="Enable SSL connection.",
        default=True,
    )
