from typing import List

from fastapi import APIRouter, Depends
from fastapi_pagination import Page, Params
from llama_index.core.base.llms.types import ChatMessage
from pydantic import BaseModel

from app.api.deps import CurrentSuperuserDep, SessionDep
from app.logger import logger
from app.models import AdminLLM, LLM, LLMUpdate
from app.rag.llms.provider import LLMProviderOption, llm_provider_options
from app.rag.llms.resolver import resolve_llm
from app.repositories.llm import llm_repo


router = APIRouter()


@router.get("/admin/llms/providers/options")
def list_llm_provider_options(user: CurrentSuperuserDep) -> List[LLMProviderOption]:
    return llm_provider_options


@router.get("/admin/llms")
def list_llms(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> Page[AdminLLM]:
    return llm_repo.paginate(db_session, params)


class LLMTestResult(BaseModel):
    success: bool
    error: str = ""


@router.post("/admin/llms/test")
def test_llm(
    db_llm: LLM,
    user: CurrentSuperuserDep,
) -> LLMTestResult:
    try:
        llm = resolve_llm(
            provider=db_llm.provider,
            model=db_llm.model,
            config=db_llm.config,
            credentials=db_llm.credentials,
        )
        llm.chat([ChatMessage(role="user", content="Who are you?")])

        # Test with dspy LM.
        import dspy
        from app.rag.llms.dspy import get_dspy_lm_by_llama_llm

        dspy_lm = get_dspy_lm_by_llama_llm(llm)
        with dspy.context(lm=dspy_lm):
            math = dspy.Predict("question -> answer: float")
            prediction = math(question="1 + 1 = ?")
            assert prediction.answer == 2

        success = True
        error = ""
    except Exception as e:
        logger.info(f"Failed to test LLM: {e}")
        success = False
        error = str(e)
    return LLMTestResult(success=success, error=error)


@router.post("/admin/llms")
def create_llm(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    llm: LLM,
) -> AdminLLM:
    return llm_repo.create(db_session, llm)


@router.get("/admin/llms/{llm_id}")
def get_llm(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    llm_id: int,
) -> AdminLLM:
    return llm_repo.must_get(db_session, llm_id)


@router.put("/admin/llms/{llm_id}")
def update_llm(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    llm_id: int,
    llm_update: LLMUpdate,
) -> AdminLLM:
    llm = llm_repo.must_get(db_session, llm_id)
    return llm_repo.update(db_session, llm, llm_update)


@router.delete("/admin/llms/{llm_id}")
def delete_llm(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    llm_id: int,
) -> None:
    llm = llm_repo.must_get(db_session, llm_id)
    llm_repo.delete(db_session, llm)


@router.put("/admin/llms/{llm_id}/set_default")
def set_default_llm(
    db_session: SessionDep, user: CurrentSuperuserDep, llm_id: int
) -> AdminLLM:
    llm = llm_repo.must_get(db_session, llm_id)
    return llm_repo.set_default(db_session, llm)
