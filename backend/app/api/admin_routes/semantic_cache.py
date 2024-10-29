from typing import Optional, Dict
import time
import logging

from fastapi import APIRouter, Body
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.chat_config import ChatEngineConfig
from app.rag.semantic_cache import SemanticCacheManager, SemanticItem

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/admin/semantic_cache")
async def add_semantic_cache(
    session: SessionDep,
    user: CurrentSuperuserDep,
    question: str,
    answer: str,
    chat_id: str,
    user_message_id: int,
    assistant_message_id: int,
    namespace: str = "default",
    chat_engine: str = "default",
    metadata: Optional[dict] = Body(None),
) -> Dict:
    chat_engine_config = ChatEngineConfig.load_from_db(session, chat_engine)
    _dspy_lm = chat_engine_config.get_dspy_lm(session)

    scm = SemanticCacheManager(
        dspy_llm=_dspy_lm,
    )

    # fill the chat related information into metadata
    if metadata is None:
        metadata = {}

    metadata["chat_detail"] = {
        "chat_id": chat_id,
        "user_message_id": user_message_id,
        "assistant_message_id": assistant_message_id,
    }

    try:
        scm.add_cache(
            session,
            item=SemanticItem(question=question, answer=answer),
            namespace=namespace,
            metadata=metadata,
        )
    except Exception as e:
        return {
            "status": "failed",
            "message": str(e),
        }

    return {
        "status": "success",
    }


@router.get("/admin/semantic_cache")
async def search_semantic_cache(
    session: SessionDep,
    user: CurrentSuperuserDep,
    query: str,
    namespace: str = "default",
    chat_engine: str = "default",
) -> Dict:
    start_time = time.time()
    chat_engine_config = ChatEngineConfig.load_from_db(session, chat_engine)
    _dspy_lm = chat_engine_config.get_dspy_lm(session)
    logger.debug(
        f"[search_semantic_cache] Loading dspy_lm took {time.time() - start_time:.2f} seconds"
    )

    scm = SemanticCacheManager(
        dspy_llm=_dspy_lm,
    )

    start_time = time.time()
    response = scm.search(
        session=session,
        query=query,
        namespace=namespace,
    )
    logger.debug(
        f"[search_semantic_cache] Searching semantic cache took {time.time() - start_time:.2f} seconds"
    )
    return response
