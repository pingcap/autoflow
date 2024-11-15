from pydantic import BaseModel, field_validator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi_pagination import Params, Page

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.models import (
    DataSource,
    DataSourceType,
)
from app.tasks import (
    import_documents_from_datasource,
    purge_datasource_related_resources,
    build_kg_index_from_chunk,
    build_vector_index_from_document,
)
from app.repositories import data_source_repo
from app.api.admin_routes.knowledge_base.models import VectorIndexError, KGIndexError

router = APIRouter()


class DataSourceCreate(BaseModel):
    name: str
    description: str
    data_source_type: DataSourceType
    config: dict | list
    build_kg_index: bool = False
    llm_id: int | None = None

    @field_validator("name")
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Please provide a name for the data source")
        return v


@router.post("/admin/datasources")
def create_datasource(
    session: SessionDep, user: CurrentSuperuserDep, request: DataSourceCreate
) -> DataSource:
    data_source = DataSource(
        name=request.name,
        description=request.description,
        data_source_type=request.data_source_type,
        config=request.config,
        build_kg_index=request.build_kg_index,
        user_id=user.id,
        llm_id=request.llm_id,
    )
    data_source = data_source_repo.create(session, data_source)
    import_documents_from_datasource.delay(data_source.id)
    return data_source


@router.get("/admin/datasources")
def list_datasources(
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> Page[DataSource]:
    return data_source_repo.paginate(session, params)


@router.get("/admin/datasources/{data_source_id}")
def get_datasource(
    session: SessionDep,
    user: CurrentSuperuserDep,
    data_source_id: int,
) -> DataSource:
    data_source = data_source_repo.get(session, data_source_id)
    if data_source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        )
    return data_source


@router.delete("/admin/datasources/{data_source_id}")
def delete_datasource(
    session: SessionDep,
    user: CurrentSuperuserDep,
    data_source_id: int,
):
    data_source = data_source_repo.get(session, data_source_id)
    if data_source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        )
    data_source_repo.delete(session, data_source)
    purge_datasource_related_resources.apply_async(args=[data_source_id], countdown=5)


@router.get("/admin/datasources/{data_source_id}/overview", deprecated=True)
def get_datasource_overview(
    session: SessionDep,
    user: CurrentSuperuserDep,
    data_source_id: int,
) -> dict:
    data_source = data_source_repo.get(session, data_source_id)
    if data_source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        )
    return data_source_repo.overview(session, data_source)


@router.get("/admin/datasources/{data_source_id}/vector-index-errors", deprecated=True)
def get_datasource_vector_index_errors(
    session: SessionDep,
    user: CurrentSuperuserDep,
    data_source_id: int,
    params: Params = Depends(),
) -> Page[VectorIndexError]:
    data_source = data_source_repo.get(session, data_source_id)
    if data_source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return data_source_repo.vector_index_built_errors(session, data_source, params)


@router.get("/admin/datasources/{data_source_id}/kg-index-errors", deprecated=True)
def get_datasource_kg_index_errors(
    session: SessionDep,
    user: CurrentSuperuserDep,
    data_source_id: int,
    params: Params = Depends(),
) -> Page[KGIndexError]:
    data_source = data_source_repo.get(session, data_source_id)
    if data_source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return data_source_repo.kg_index_built_errors(session, data_source, params)


@router.post("/admin/datasources/{data_source_id}/retry-failed-tasks", deprecated=True)
def retry_failed_tasks(
    session: SessionDep,
    user: CurrentSuperuserDep,
    data_source_id: int,
) -> None:
    data_source = data_source_repo.get(session, data_source_id)
    if data_source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    for docuemnt_id in data_source_repo.set_failed_vector_index_tasks_to_pending(
        session, data_source
    ):
        build_vector_index_from_document.delay(data_source_id, docuemnt_id)
    for chunk_id, document_id in data_source_repo.set_failed_kg_index_tasks_to_pending(
        session, data_source
    ):
        build_kg_index_from_chunk.delay(data_source_id, document_id, chunk_id)
    return
