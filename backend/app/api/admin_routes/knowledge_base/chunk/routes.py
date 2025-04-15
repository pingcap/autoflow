import logging

from fastapi import APIRouter, HTTPException
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.retrievers.chunk.simple_retriever import (
    ChunkSimpleRetriever,
)
from app.rag.retrievers.chunk.schema import ChunksRetrievalResult

from app.exceptions import InternalServerError
from .models import KBRetrieveChunksRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/admin/knowledge_base/{kb_id}/chunks/retrieve")
def retrieve_chunks(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    request: KBRetrieveChunksRequest,
) -> ChunksRetrievalResult:
    try:
        vector_search_config = request.retrieval_config.vector_search
        retriever = ChunkSimpleRetriever(
            db_session=db_session,
            knowledge_base_id=kb_id,
            config=vector_search_config,
        )
        return retriever.retrieve_chunks(
            request.query,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
