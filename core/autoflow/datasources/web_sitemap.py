import logging
from typing import Generator
from urllib.parse import urlparse, urljoin

import requests
from pydantic import BaseModel
from bs4 import BeautifulSoup

from autoflow.datasources.base import DataSource
from autoflow.datasources.web_base import load_web_documents
from autoflow.db_models import DBDocument

logger = logging.getLogger(__name__)


class WebSitemapDataSourceConfig(BaseModel):
    url: str


def _ensure_absolute_url(source_url: str, maybe_relative_url: str) -> str:
    if not urlparse(maybe_relative_url).netloc:
        return urljoin(source_url, maybe_relative_url)
    return maybe_relative_url


def extract_urls_from_sitemap(sitemap_url: str) -> list[str]:
    response = requests.get(sitemap_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    result = [
        _ensure_absolute_url(sitemap_url, loc_tag.text)
        for loc_tag in soup.find_all("loc")
    ]
    if not result:
        raise ValueError(f"No URLs found in sitemap {sitemap_url}")
    return result


class WebSitemapDataSource(DataSource):
    def validate_config(self, config: dict):
        return WebSitemapDataSourceConfig.model_validate(config)

    def load_documents(self) -> Generator[DBDocument, None, None]:
        sitemap_url = self.config["url"]
        urls = extract_urls_from_sitemap(sitemap_url)
        logger.info(f"Found {len(urls)} URLs in sitemap {sitemap_url}")
        yield from load_web_documents(self.knowledge_base_id, self.data_source_id, urls)
