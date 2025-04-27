import logging

from fastapi import APIRouter
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.retrievers.knowledge_graph.fusion_retriever import (
    KnowledgeGraphFusionRetriever,
)
from app.rag.retrievers.knowledge_graph.schema import (
    KnowledgeGraphRetrievalResult,
)
from app.rag.retrievers.chunk.fusion_retriever import (
    KBChunkRetriever,
)
from app.exceptions import KBNotFound
from app.rag.retrievers.chunk.schema import ChunksRetrievalResult
from app.rag.llms.resolver import get_llm_or_default
from .models import ChunksRetrievalRequest, KnowledgeGraphRetrievalRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/retrieve/chunks")
def retrieve_chunks(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    request: ChunksRetrievalRequest,
) -> ChunksRetrievalResult:
    try:
        config = request.retrieval_config
        llm = get_llm_or_default(db_session, config.llm_id)
        retriever = KBChunkRetriever(
            db_session=db_session,
            knowledge_base_ids=config.knowledge_base_ids,
            llm=llm,
            use_query_decompose=config.use_query_decompose,
            kb_select_mode=config.kb_select_mode,
            config=config.vector_search,
        )
        return retriever.retrieve_chunks(request.query, config.full_documents)
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise


@router.post("/retrieve/knowledge_graph")
def retrieve_knowledge_graph(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    request: KnowledgeGraphRetrievalRequest,
) -> KnowledgeGraphRetrievalResult:
    try:
        config = request.retrieval_config
        llm = get_llm_or_default(db_session, config.llm_id)
        retriever = KnowledgeGraphFusionRetriever(
            db_session=db_session,
            knowledge_base_ids=config.knowledge_base_ids,
            llm=llm,
            use_query_decompose=config.use_query_decompose,
            kb_select_mode=config.kb_select_mode,
            config=config.knowledge_graph,
        )
        return retriever.retrieve_knowledge_graph(request.query)
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise
