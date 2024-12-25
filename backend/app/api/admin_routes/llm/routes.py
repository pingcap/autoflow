import logging
from typing import List

from fastapi import APIRouter, Depends
from fastapi_pagination import Params, Page
from pydantic import BaseModel
from sqlalchemy import update

from app.api.deps import CurrentSuperuserDep, SessionDep
from app.exceptions import InternalServerError, LLMNotFound
from app.models import AdminLLM, LLM, ChatEngine, DataSource, KnowledgeBase
from app.rag.chat_config import get_llm
from app.rag.llm_option import LLMOption, admin_llm_options
from app.repositories.llm import llm_repo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/admin/llms/options")
def get_llm_options(user: CurrentSuperuserDep) -> List[LLMOption]:
    return admin_llm_options


@router.get("/admin/llms")
def list_llms(
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> Page[AdminLLM]:
    return llm_repo.paginate(session, params)


@router.post("/admin/llms")
def create_llm(
    llm: LLM,
    session: SessionDep,
    user: CurrentSuperuserDep,
) -> AdminLLM:
    return llm_repo.create(session, llm)


class LLMTestResult(BaseModel):
    success: bool
    error: str = ""


@router.post("/admin/llms/test")
def test_llm(
    db_llm: LLM,
    user: CurrentSuperuserDep,
) -> LLMTestResult:
    try:
        llm = get_llm(
            provider=db_llm.provider,
            model=db_llm.model,
            config=db_llm.config,
            credentials=db_llm.credentials,
        )
        llm.complete("Who are you?")
        success = True
        error = ""
    except Exception as e:
        logger.debug(e)
        success = False
        error = str(e)
    return LLMTestResult(success=success, error=error)


@router.get("/admin/llms/{llm_id}")
def get_llm_detail(
    session: SessionDep,
    user: CurrentSuperuserDep,
    llm_id: int,
) -> AdminLLM:
    try:
        return llm_repo.must_get(session, llm_id)
    except LLMNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.delete("/admin/llms/{llm_id}")
def delete_llm(
    llm_id: int,
    session: SessionDep,
    user: CurrentSuperuserDep,
) -> AdminLLM:
    llm = llm_repo.must_get(session, llm_id)

    # FIXME: Should be replaced with a new LLM or prohibit users from operating,
    #  If the current LLM is used by a Chat Engine or Knowledge Base.

    session.exec(
        update(ChatEngine).where(ChatEngine.llm_id == llm_id).values(llm_id=None)
    )
    session.exec(
        update(ChatEngine)
        .where(ChatEngine.fast_llm_id == llm_id)
        .values(fast_llm_id=None)
    )
    session.exec(
        update(KnowledgeBase).where(KnowledgeBase.llm_id == llm_id).values(llm_id=None)
    )
    session.delete(llm)
    session.commit()
    return llm


@router.put("/admin/llms/{llm_id}/set_default")
def set_default_llm(
    session: SessionDep, user: CurrentSuperuserDep, llm_id: int
) -> AdminLLM:
    try:
        llm = llm_repo.must_get(session, llm_id)
        llm_repo.set_default_model(session, llm_id)
        session.refresh(llm)
        return llm
    except LLMNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
