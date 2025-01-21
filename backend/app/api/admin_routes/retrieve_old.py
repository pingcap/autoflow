import logging
from typing import Optional, List

from fastapi import APIRouter
from app.models import Document
from app.api.admin_routes.models import ChatEngineBasedRetrieveRequest
from app.api.deps import SessionDep, CurrentSuperuserDep
from llama_index.core.schema import NodeWithScore
from app.rag.retrieve import retrieve_service

from app.exceptions import InternalServerError, KBNotFound

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/admin/retrieve/documents", deprecated=True)
def retrieve_documents(
    session: SessionDep,
    user: CurrentSuperuserDep,
    question: str,
    chat_engine: str = "default",
    top_k: Optional[int] = 5,
    similarity_top_k: Optional[int] = None,
    oversampling_factor: Optional[int] = 5,
    enable_kg_enhance_query_refine: Optional[bool] = True,
) -> List[Document]:
    try:
        return retrieve_service.chat_engine_retrieve_documents(
            session,
            question=question,
            top_k=top_k,
            chat_engine_name=chat_engine,
            similarity_top_k=similarity_top_k,
            oversampling_factor=oversampling_factor,
            enable_kg_enhance_query_refine=enable_kg_enhance_query_refine,
        )
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/admin/embedding_retrieve", deprecated=True)
def embedding_retrieve(
    session: SessionDep,
    user: CurrentSuperuserDep,
    question: str,
    chat_engine: str = "default",
    top_k: Optional[int] = 5,
    similarity_top_k: Optional[int] = None,
    oversampling_factor: Optional[int] = 5,
    enable_kg_enhance_query_refine=False,
) -> List[NodeWithScore]:
    try:
        nodes = retrieve_service.chat_engine_retrieve_chunks(
            session,
            question=question,
            top_k=top_k,
            chat_engine_name=chat_engine,
            similarity_top_k=similarity_top_k,
            oversampling_factor=oversampling_factor,
            enable_kg_enhance_query_refine=enable_kg_enhance_query_refine,
        )
        return nodes
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/admin/embedding_retrieve", deprecated=True)
def embedding_search(
    session: SessionDep,
    user: CurrentSuperuserDep,
    request: ChatEngineBasedRetrieveRequest,
) -> List[NodeWithScore]:
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
