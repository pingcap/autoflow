from typing import Generator
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.models import Document
from app.rag.datasource.base import BaseDataSource
from app.rag.datasource.web_base import load_web_documents
from app.security.crawl_http import checked_url
from app.security.crawler import crawl


class WebSitemapConfig(BaseModel):
    url: str = Field(min_length=1, max_length=512)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value):
        return str(checked_url(value))


def extract_urls_from_sitemap(sitemap_url: str) -> list[str]:
    return crawl(
        sitemap_url,
        kind="sitemap",
        private_origins=settings.CRAWLER_ALLOWED_PRIVATE_ORIGINS,
    )


class WebSitemapDataSource(BaseDataSource):
    def validate_config(self):
        self.config = WebSitemapConfig.model_validate(self.config).model_dump()

    def load_documents(self) -> Generator[Document, None, None]:
        urls = extract_urls_from_sitemap(self.config["url"])
        yield from load_web_documents(self.knowledge_base_id, self.data_source_id, urls)
