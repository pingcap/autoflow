import logging
from typing import List

from fastapi import APIRouter
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.indexes.vector_search.config import VectorSearchConfig
from app.rag.indexes.vector_search.retriever import (
    RetrievedChunk,
    VectorSearchRetriever,
)

from app.exceptions import InternalServerError, KBNotFound
from .models import RetrieveChunkRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/admin/knowledge_base/{kb_id}/chunks")
def retrieve_chunks(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    request: RetrieveChunkRequest,
) -> List[RetrievedChunk]:
    try:
        retriever = VectorSearchRetriever(
            db_session=session,
            knowledge_base_id=kb_id,
            config=VectorSearchConfig(**request.vector_search_config.model_dump()),
        )
        return retriever.retrieve_chunks(request.query, session)
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
