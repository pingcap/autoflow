from enum import Enum

from llama_index.core.schema import BaseComponent


class DataSourceType(str, Enum):
    FILE = "file"
    WEB_SITEMAP = "web_sitemap"
    WEB_SINGLE_PAGE = "web_single_page"


class IndexMethod(str, Enum):
    VECTOR_SEARCH = "VECTOR_SEARCH"
    FULLTEXT_SEARCH = "FULLTEXT_SEARCH"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"


BaseComponent = BaseComponent
