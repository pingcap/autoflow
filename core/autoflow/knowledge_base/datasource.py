from typing import Any

from autoflow.datasources import (
    DataSource,
    FileDataSource,
    WebSitemapDataSource,
    WebSinglePageDataSource,
)
from autoflow.schema import DataSourceType


def get_datasource_by_type(type: DataSourceType, config: Any) -> DataSource:
    if type == DataSourceType.FILE:
        return FileDataSource(config)
    elif type == DataSourceType.WEB_SITEMAP:
        return WebSitemapDataSource(config)
    elif type == DataSourceType.WEB_SINGLE_PAGE:
        return WebSinglePageDataSource(config)
    else:
        raise ValueError(f"Unknown datasource type: {type}")
