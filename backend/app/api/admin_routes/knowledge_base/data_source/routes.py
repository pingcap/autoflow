import logging

from fastapi import APIRouter, Depends
from fastapi_pagination import Params, Page

from app.api.admin_routes.knowledge_base.data_source.models import (
    KBDataSourceUpdate,
    KBDataSource,
)
from app.api.admin_routes.knowledge_base.models import KBDataSourceCreate
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.exceptions import InternalServerError, KBDataSourceNotFound, KBNotFound
from app.models import DataSource
from app.repositories import knowledge_base_repo
from app.tasks.knowledge_base import (
    import_documents_from_kb_datasource,
    purge_kb_datasource_related_resources,
)


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/admin/knowledge_bases/{kb_id}/datasources")
def create_kb_datasource(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    create: KBDataSourceCreate,
) -> KBDataSource:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        new_data_source = DataSource(
            name=create.name,
            description="",
            data_source_type=create.data_source_type,
            config=create.config,
        )
        new_data_source = knowledge_base_repo.add_kb_datasource(
            session, kb, new_data_source
        )

        import_documents_from_kb_datasource.delay(kb_id, new_data_source.id)

        return new_data_source
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.error(
            f"Failed to create data source for knowledge base #{kb_id}: {e}", exc_info=e
        )
        raise InternalServerError()


@router.put("/admin/knowledge_bases/{kb_id}/datasources/{data_source_id}")
def update_kb_datasource(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    data_source_id: int,
    update: KBDataSourceUpdate,
) -> KBDataSource:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)

        data_source = kb.must_get_data_source_by_id(data_source_id)
        data_source.name = update.name

        session.add(data_source)
        session.commit()
        session.refresh(data_source)

        return data_source
    except KBNotFound as e:
        raise e
    except KBDataSourceNotFound as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to update data source #{data_source_id}: {e}", exc_info=e)
        raise InternalServerError()


@router.get("/admin/knowledge_bases/{kb_id}/datasources/{data_source_id}")
def get_kb_datasource(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    data_source_id: int,
) -> KBDataSource:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        return kb.must_get_data_source_by_id(data_source_id)
    except KBNotFound as e:
        raise e
    except KBDataSourceNotFound as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to get data source #{data_source_id}: {e}", exc_info=e)
        raise InternalServerError()


@router.get("/admin/knowledge_bases/{kb_id}/datasources")
def list_kb_datasources(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    params: Params = Depends(),
) -> Page[KBDataSource]:
    return knowledge_base_repo.list_kb_datasources(session, kb_id, params)


@router.delete("/admin/knowledge_bases/{kb_id}/datasources/{data_source_id}")
def remove_kb_datasource(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    data_source_id: int,
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        data_source = kb.must_get_data_source_by_id(data_source_id)

        # Flag the data source to be deleted, it will be deleted completely by the background job.
        knowledge_base_repo.remove_kb_datasource(session, kb, data_source)
        session.commit()

        purge_kb_datasource_related_resources.apply_async(
            args=[kb_id, data_source_id], countdown=5
        )

        return {"detail": "success"}
    except KBNotFound as e:
        raise e
    except KBDataSourceNotFound as e:
        raise e
    except Exception as e:
        logger.error(
            f"Failed to remove data source #{data_source_id} from knowledge base #{kb_id}: {e}",
            exc_info=e,
        )
        raise InternalServerError()
