from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.api.admin_routes.models import DataSourceDescriptor, KnowledgeBaseDescriptor
from app.models import DocIndexTaskStatus
from app.types import MimeTypes


class DocumentFilters(BaseModel):
    name: Optional[str] = Field(
        None,
        description="[Fuzzy Match] name field, will search for the name that contains the given string.",
    )
    source_uri: Optional[str] = Field(
        None,
        description="[Fuzzy Match] source URI field, will search for the source URI that contains the given string.",
    )
    knowledge_base_id: Optional[int] = Field(None)
    data_source_id: Optional[int] = Field(None)
    created_at_start: Optional[datetime] = Field(None)
    created_at_end: Optional[datetime] = None
    updated_at_start: Optional[datetime] = None
    updated_at_end: Optional[datetime] = None
    last_modified_at_start: Optional[datetime] = None
    last_modified_at_end: Optional[datetime] = None
    mime_type: Optional[MimeTypes] = None
    index_status: Optional[DocIndexTaskStatus] = None


class DocumentItem(BaseModel):
    id: int
    hash: str
    name: str
    content: str
    mime_type: MimeTypes | None
    source_uri: str | None
    meta: dict | list | None
    index_status: DocIndexTaskStatus | None
    index_result: str | None
    data_source: DataSourceDescriptor | None
    knowledge_base: KnowledgeBaseDescriptor | None
    last_modified_at: datetime
    created_at: datetime
    updated_at: datetime
