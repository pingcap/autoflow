import logging

from fastapi import APIRouter
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.indices.vector_search.retriever.simple_retriever import (
    VectorSearchSimpleRetriever,
)
from app.rag.indices.vector_search.retriever.schema import ChunksRetrievalResult

from app.exceptions import InternalServerError, KBNotFound
from .models import KBRetrieveChunksRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/admin/knowledge_base/{kb_id}/chunks/retrieve")
def retrieve_chunks(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    request: KBRetrieveChunksRequest,
) -> ChunksRetrievalResult:
    try:
        vector_search_config = request.retrieval_config.vector_search
        retriever = VectorSearchSimpleRetriever(
            db_session=db_session,
            knowledge_base_id=kb_id,
            config=vector_search_config,
        )
        return retriever.retrieve_chunks(
            request.query,
        )
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
