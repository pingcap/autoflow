import logging

from llama_index.core.base.embeddings.base import BaseEmbedding
from sqlmodel import Session

from app.models.knowledge_base import KnowledgeBase
from app.rag.chat_config import (
    get_default_llm,
    get_llm,
    get_embed_model,
    get_default_embed_model,
)
from app.utils.dspy import get_dspy_lm_by_llama_llm


logger = logging.getLogger(__name__)


def get_kb_llm(session: Session, kb: KnowledgeBase):
    db_llm = kb.llm
    if db_llm:
        return get_llm(db_llm.provider, db_llm.model, db_llm.config, db_llm.credentials)
    else:
        return get_default_llm(session)


def get_kb_dspy_llm(session: Session, kb: KnowledgeBase):
    llm = get_kb_llm(session, kb)
    return get_dspy_lm_by_llama_llm(llm)


def get_kb_embed_model(session: Session, kb: KnowledgeBase) -> BaseEmbedding:
    db_embed_model = kb.embedding_model
    if db_embed_model:
        return get_embed_model(
            db_embed_model.provider,
            db_embed_model.model,
            db_embed_model.config,
            db_embed_model.credentials,
        )
    else:
        return get_default_embed_model(session)
