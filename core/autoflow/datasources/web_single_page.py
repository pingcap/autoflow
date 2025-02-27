import logging
from pydantic import BaseModel
from typing import Generator, List

from autoflow.datasources import DataSource
from autoflow.datasources.web_base import load_web_documents
from autoflow.db_models import DBDocument

logger = logging.getLogger(__name__)


class WebSinglePageDataSourceConfig(BaseModel):
    urls: List[str]


class WebSinglePageDataSource(DataSource[WebSinglePageDataSourceConfig]):
    def validate_config(self, config: dict):
        return WebSinglePageDataSourceConfig.model_validate(config)

    def load_documents(self) -> Generator[DBDocument, None, None]:
        yield from load_web_documents(self.config.urls)
