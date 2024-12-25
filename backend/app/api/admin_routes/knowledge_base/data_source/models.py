from pydantic import BaseModel, field_validator

from app.models import DataSourceType


class KBDataSource(BaseModel):
    """
    Represents a linked data source for a knowledge base.
    """

    id: int
    name: str
    data_source_type: DataSourceType
    config: dict | list


class KBDataSourceMutable(BaseModel):
    name: str

    @field_validator("name")
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Please provide a name for the data source")
        return v


class KBDataSourceCreate(KBDataSourceMutable):
    data_source_type: DataSourceType
    config: dict | list


class KBDataSourceUpdate(KBDataSourceMutable):
    pass
