from fastapi import APIRouter
from sqlmodel import text

from app.api.deps import SessionDep
from app.api.routes.models import SystemConfigStatusResponse
from app.site_settings import SiteSetting
from app.rag.chat import (
    check_rag_required_config,
    check_rag_optional_config,
    check_rag_config_need_migration,
)

router = APIRouter()


@router.get("/healthz")
def status(session: SessionDep):
    now = session.exec(text("SELECT NOW()")).scalar()
    return {"now": now}


@router.get("/site-config")
def site_config() -> dict:
    return SiteSetting.get_client_settings()


@router.get("/system/bootstrap-status")
def system_bootstrap_status(session: SessionDep) -> SystemConfigStatusResponse:
    required_config_check_status = check_rag_required_config(session)
    optional_config_check_status = check_rag_optional_config(session)
    need_migration_status = check_rag_config_need_migration(session)

    return SystemConfigStatusResponse(
        required=required_config_check_status,
        optional=optional_config_check_status,
        need_migration=need_migration_status,
    )
