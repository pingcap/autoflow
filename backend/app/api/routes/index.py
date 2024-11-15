from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import text

from app.api.deps import SessionDep
from app.site_settings import SiteSetting
from app.rag.chat import check_rag_required_config, check_rag_optional_config

router = APIRouter()


@router.get("/healthz")
def status(session: SessionDep):
    now = session.exec(text("SELECT NOW()")).scalar()
    return {"now": now}


@router.get("/site-config")
def site_config() -> dict:
    return SiteSetting.get_client_settings()


class RequiredConfigStatus(BaseModel):
    default_llm: bool
    default_embedding_model: bool
    datasource: bool
    knowledge_base: bool


class OptionalConfigStatus(BaseModel):
    langfuse: bool
    default_reranker: bool


class SystemConfigStatusResponse(BaseModel):
    required: RequiredConfigStatus
    optional: OptionalConfigStatus


@router.get("/system/bootstrap-status")
def system_bootstrap_status(session: SessionDep) -> SystemConfigStatusResponse:
    has_default_llm, has_default_embedding_model, has_datasource, has_knowledge_base = (
        check_rag_required_config(session)
    )
    langfuse, default_reranker = check_rag_optional_config(session)
    return SystemConfigStatusResponse(
        required=RequiredConfigStatus(
            default_llm=has_default_llm,
            default_embedding_model=has_default_embedding_model,
            datasource=has_datasource,
            knowledge_base=has_knowledge_base
        ),
        optional=OptionalConfigStatus(
            langfuse=langfuse,
            default_reranker=default_reranker,
        ),
    )
