import logging
from typing import Optional, List

from fastapi import APIRouter
from sqlmodel import Session
from app.models import Document
from app.api.admin_routes.models import ChatEngineBasedRetrieveRequest
from app.api.deps import SessionDep, CurrentSuperuserDep
from llama_index.core.schema import NodeWithScore

from app.exceptions import InternalServerError, KBNotFound
from app.rag.chat.config import ChatEngineConfig
from app.rag.chat.retrieve.retrieve_flow import RetrieveFlow

router = APIRouter()
logger = logging.getLogger(__name__)


def get_override_engine_config(
    db_session: Session,
    engine_name: str,
    # Override chat engine config.
    top_k: Optional[int] = None,
    similarity_top_k: Optional[int] = None,
    oversampling_factor: Optional[int] = None,
    refine_question_with_kg: Optional[bool] = None,
) -> ChatEngineConfig:
    engine_config = ChatEngineConfig.load_from_db(db_session, engine_name)
    if similarity_top_k is not None:
        engine_config.vector_search.similarity_top_k = similarity_top_k
    if oversampling_factor is not None:
        engine_config.vector_search.oversampling_factor = oversampling_factor
    if top_k is not None:
        engine_config.vector_search.top_k = top_k
    if refine_question_with_kg is not None:
        engine_config.refine_question_with_kg = refine_question_with_kg
    return engine_config


@router.get("/admin/retrieve/documents", deprecated=True)
def legacy_retrieve_documents(
    session: SessionDep,
    user: CurrentSuperuserDep,
    question: str,
    chat_engine: str = "default",
    # Override chat engine config.
    top_k: Optional[int] = 5,
    similarity_top_k: Optional[int] = None,
    oversampling_factor: Optional[int] = 5,
    refine_question_with_kg: Optional[bool] = True,
) -> List[Document]:
    try:
        engine_config = get_override_engine_config(
            db_session=session,
            engine_name=chat_engine,
            top_k=top_k,
            similarity_top_k=similarity_top_k,
            oversampling_factor=oversampling_factor,
            refine_question_with_kg=refine_question_with_kg,
        )
        retriever = RetrieveFlow(
            db_session=session,
            engine_name=chat_engine,
            engine_config=engine_config,
        )
        return retriever.retrieve_documents(question)
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/admin/embedding_retrieve", deprecated=True)
def legacy_retrieve_chunks(
    session: SessionDep,
    user: CurrentSuperuserDep,
    question: str,
    chat_engine: str = "default",
    # Override chat engine config.
    top_k: Optional[int] = 5,
    similarity_top_k: Optional[int] = None,
    oversampling_factor: Optional[int] = 5,
    refine_question_with_kg=False,
) -> List[NodeWithScore]:
    try:
        engine_config = get_override_engine_config(
            db_session=session,
            engine_name=chat_engine,
            top_k=top_k,
            similarity_top_k=similarity_top_k,
            oversampling_factor=oversampling_factor,
            refine_question_with_kg=refine_question_with_kg,
        )
        retriever = RetrieveFlow(
            db_session=session,
            engine_name=chat_engine,
            engine_config=engine_config,
        )
        return retriever.retrieve(question)
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/admin/embedding_retrieve", deprecated=True)
def legacy_retrieve_chunks_2(
    session: SessionDep,
    user: CurrentSuperuserDep,
    request: ChatEngineBasedRetrieveRequest,
) -> List[NodeWithScore]:
    try:
        engine_config = get_override_engine_config(
            db_session=session,
            engine_name=request.chat_engine,
            top_k=request.top_k,
            similarity_top_k=request.similarity_top_k,
            oversampling_factor=request.oversampling_factor,
            refine_question_with_kg=request.refine_question_with_kg,
        )
        retriever = RetrieveFlow(
            db_session=session,
            engine_name=request.chat_engine,
            engine_config=engine_config,
        )
        return retriever.retrieve(request.query)
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
