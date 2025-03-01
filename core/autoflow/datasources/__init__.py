from .base import DataSource
from .file import FileDataSource
from .web_sitemap import WebSitemapDataSource
from .web_single_page import WebSinglePageDataSource

__all__ = [
    "DataSource",
    "FileDataSource",
    "WebSitemapDataSource",
    "WebSinglePageDataSource",
]
