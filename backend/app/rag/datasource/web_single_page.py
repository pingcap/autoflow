from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Generator

from app.models import Document
from app.rag.datasource.base import BaseDataSource
from app.rag.datasource.web_base import load_web_documents
from app.security.crawl_http import MAX_URLS, checked_url


class WebSinglePageConfig(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=MAX_URLS)

    @model_validator(mode="before")
    @classmethod
    def legacy_single_url(cls, value):
        if isinstance(value, dict) and "urls" not in value and "url" in value:
            return {**value, "urls": [value["url"]]}
        return value

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, values):
        for value in values:
            if len(value) > 512:
                raise ValueError("Document URL is too long")
            checked_url(value)
        return values


class WebSinglePageDataSource(BaseDataSource):
    def validate_config(self):
        self.config = WebSinglePageConfig.model_validate(self.config).model_dump()

    def load_documents(self) -> Generator[Document, None, None]:
        yield from load_web_documents(
            self.knowledge_base_id, self.data_source_id, self.config["urls"]
        )
