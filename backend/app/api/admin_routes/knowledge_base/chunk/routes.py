import logging

from fastapi import APIRouter
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.retrieve import retrieve_service

from app.exceptions import InternalServerError, KBNotFound
from .models import RetrieveRequest, RetrieveResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/admin/retrieve/chunks")
def retrieve_chunks(
    session: SessionDep,
    user: CurrentSuperuserDep,
    request: RetrieveRequest,
) -> RetrieveResponse:
    try:
        return retrieve_service.chat_engine_retrieve_chunks(
            session,
            request.query,
            top_k=request.top_k,
            similarity_top_k=request.similarity_top_k,
            oversampling_factor=request.oversampling_factor,
            enable_kg_enhance_query_refine=request.enable_kg_enhance_query_refine,
        )
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
