import logging

from typing import List
from fastapi import APIRouter
from llama_index.core import QueryBundle
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.indices.vector_search.base_retriever import (
    VectorSearchRetriever,
)
from app.rag.indices.vector_search.schema import RetrievedChunk

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
) -> List[RetrievedChunk]:
    try:
        # TODO: support knowledge graph search.
        vector_search_config = request.retrieval_config.vector_search
        retriever = VectorSearchRetriever(
            db_session=db_session,
            knowledge_base_id=kb_id,
            config=vector_search_config,
        )
        return retriever.retrieve_chunks(QueryBundle(request.query), db_session)
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
