from pydantic import BaseModel

from autoflow.configs.db import DatabaseConfig


class Config(BaseModel):
    db: DatabaseConfig = DatabaseConfig()
