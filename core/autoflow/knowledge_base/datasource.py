from typing import Any

from autoflow.datasources import (
    DataSource,
    FileDataSource,
    WebSitemapDataSource,
    WebSinglePageDataSource,
)
from autoflow.schema import DataSourceKind


def get_datasource_by_kind(kind: DataSourceKind, config: Any) -> DataSource:
    if kind == DataSourceKind.FILE:
        return FileDataSource(config)
    elif kind == DataSourceKind.WEB_SITEMAP:
        return WebSitemapDataSource(config)
    elif kind == DataSourceKind.WEB_SINGLE_PAGE:
        return WebSinglePageDataSource(config)
    else:
        raise ValueError(f"Unknown datasource kind: {kind}")
