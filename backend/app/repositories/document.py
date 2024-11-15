from sqlmodel import select, Session, col
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate

from app.api.admin_routes.document.models import DocumentFilters
from app.models import Document
from app.repositories.base_repo import BaseRepo


class DocumentRepo(BaseRepo):
    model_cls = Document

    def paginate(
        self,
        session: Session,
        filters: DocumentFilters,
        params: Params | None = Params()
    ) -> Page[Document]:
        # build the select statement via conditions
        stmt = select(Document)
        if filters.knowledge_base_id:
            stmt = stmt.where(Document.knowledge_base_id == filters.knowledge_base_id)
        if filters.source_uri:
            stmt = stmt.where(col(Document.source_uri).contains(filters.source_uri))
        if filters.data_source_id:
            stmt = stmt.where(Document.data_source_id == filters.data_source_id)
        if filters.created_at_start:
            stmt = stmt.where(Document.created_at >= filters.created_at_start)
        if filters.created_at_end:
            stmt = stmt.where(Document.created_at <= filters.created_at_end)
        if filters.updated_at_start:
            stmt = stmt.where(Document.updated_at >= filters.updated_at_start)
        if filters.updated_at_end:
            stmt = stmt.where(Document.updated_at <= filters.updated_at_end)
        if filters.last_modified_at_start:
            stmt = stmt.where(Document.last_modified_at >= filters.last_modified_at_start)
        if filters.last_modified_at_end:
            stmt = stmt.where(Document.last_modified_at <= filters.last_modified_at_end)
        if filters.name:
            stmt = stmt.where(col(Document.name).contains(filters.name))
        if filters.mime_type:
            stmt = stmt.where(Document.mime_type == filters.mime_type)
        if filters.index_status:
            stmt = stmt.where(Document.index_status == filters.index_status)

        # Make sure the newer edited record is always on top
        stmt = stmt.order_by(Document.updated_at.desc())
        return paginate(session, stmt, params)


document_repo = DocumentRepo()
