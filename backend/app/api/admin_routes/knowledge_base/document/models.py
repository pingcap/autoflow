from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.api.admin_routes.models import DataSourceDescriptor, KnowledgeBaseDescriptor
from app.models import DocIndexTaskStatus
from app.types import MimeTypes


class DocumentFilters(BaseModel):
    search: Optional[str] = Field(
        description="The search string to filter documents by name or source URI.",
        default=None,
    )
    knowledge_base_id: Optional[int] = Field(
        description="The knowledge base ID that the document belongs to.",
        default=None,
    )
    data_source_id: Optional[int] = Field(
        description="The data source ID that the document belongs to.",
        default=None,
    )
    mime_type: Optional[MimeTypes] = Field(
        description="The MIME type of the documents to filter by.",
        default=None,
    )
    index_status: Optional[DocIndexTaskStatus] = Field(
        description="The status of the document index task to filter by.",
        default=None,
    )
    created_at: Optional[tuple[datetime, datetime]] = Field(
        description="The time range when the document was created.",
        default=None,
    )
    updated_at: Optional[tuple[datetime, datetime]] = Field(
        description="The time range when the document was last updated.",
        default=None,
    )
    last_modified_at: Optional[tuple[datetime, datetime]] = Field(
        description="The time range when the document was last modified in the source system.",
        default=None,
    )


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


class RebuildIndexResult(BaseModel):
    reindex_document_ids: list[int] = Field(default_factory=list)
    ignore_document_ids: list[int] = Field(default_factory=list)
    reindex_chunk_ids: list[UUID] = Field(default_factory=list)
    ignore_chunk_ids: list[UUID] = Field(default_factory=list)
