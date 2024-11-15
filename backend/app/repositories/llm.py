from typing import Type

from sqlmodel import select, Session

from app.exceptions import DBLLMNotFoundError
from app.models import (
    LLM as DBLLM
)


def get_db_llm(session: Session, llm_id: int) -> Type[DBLLM] | None:
    return session.get(DBLLM, llm_id)


def must_get_llm(session: Session, llm_id: int) -> Type[DBLLM]:
    db_llm = get_db_llm(session, llm_id)
    if db_llm is None:
        raise DBLLMNotFoundError(llm_id)
    return db_llm


def get_default_db_llm(session: Session) -> Type[DBLLM] | None:
    stmt = select(DBLLM).where(DBLLM.is_default == True).order_by(DBLLM.updated_at.desc()).limit(1)
    return session.exec(stmt).first()
