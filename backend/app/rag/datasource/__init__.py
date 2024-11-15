from sqlmodel import Session
from typing import Any
from uuid import UUID

from app.models import DataSourceType
from .base import BaseDataSource
from .file import FileDataSource
from .web_sitemap import WebSitemapDataSource
from .web_single_page import WebSinglePageDataSource


def get_data_source_loader(
    session: Session,
    knowledge_base_id: int,
    data_source_type: DataSourceType,
    data_source_id: int,
    user_id: UUID,
    config: Any,
) -> BaseDataSource:
    data_source_cls = None

    match data_source_type:
        case DataSourceType.FILE:
            data_source_cls = FileDataSource
        case DataSourceType.WEB_SITEMAP:
            data_source_cls = WebSitemapDataSource
        case DataSourceType.WEB_SINGLE_PAGE:
            data_source_cls = WebSinglePageDataSource
        case _:
            raise ValueError("Data source type not supported")

    return data_source_cls(session, knowledge_base_id, data_source_id, user_id, config)
