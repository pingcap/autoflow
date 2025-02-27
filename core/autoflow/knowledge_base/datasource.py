from typing import Any, Dict

from pydantic import field_validator, BaseModel

from autoflow.datasources import (
    DataSource,
    FileDataSource,
    WebSitemapDataSource,
    WebSinglePageDataSource,
)
from autoflow.db_models import DataSourceKind


class DataSourceMutable(BaseModel):
    name: str

    @field_validator("name")
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Please provide a name for the data source")
        return v


class DataSourceCreate(DataSourceMutable):
    kind: DataSourceKind
    config: Dict


DataSourceUpdate = DataSourceCreate


def get_datasource_by_kind(kind: DataSourceKind, config: Any) -> DataSource:
    if kind == DataSourceKind.FILE:
        return FileDataSource(config)
    elif kind == DataSourceKind.WEB_SITEMAP:
        return WebSitemapDataSource(config)
    elif kind == DataSourceKind.WEB_SINGLE_PAGE:
        return WebSinglePageDataSource(config)
    else:
        raise ValueError(f"Unknown datasource kind: {kind}")
