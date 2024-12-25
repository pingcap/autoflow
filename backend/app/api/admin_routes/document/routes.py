from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi_pagination import Params, Page

from app.api.admin_routes.knowledge_base.document.models import (
    DocumentFilters,
    DocumentItem,
)
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.repositories import document_repo

router = APIRouter()


@router.get("/admin/documents")
def list_documents(
    session: SessionDep,
    user: CurrentSuperuserDep,
    filters: Annotated[DocumentFilters, Query()],
    params: Params = Depends(),
) -> Page[DocumentItem]:
    return document_repo.paginate(
        session=session,
        filters=filters,
        params=params,
    )
