import logging
from pydantic import BaseModel
from typing import Generator, List

from app.models import Document
from app.rag.datasource.base import BaseDataSource
from app.rag.datasource.web_base import load_web_documents

logger = logging.getLogger(__name__)


class WebSinglePageConfig(BaseModel):
    urls: List[str]


class WebSinglePageDataSource(BaseDataSource):
    def validate_config(self):
        WebSinglePageConfig.model_validate(self.config)

    def load_documents(self) -> Generator[Document, None, None]:
        if "url" in self.config:
            # TODO: remove this once we have a proper config
            urls = [self.config["url"]]
        else:
            urls = self.config["urls"]

        yield from load_web_documents(self.knowledge_base_id, self.data_source_id, urls)
