import logging
from typing import List

from fastapi import APIRouter
from llama_index.core import QueryBundle
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.indices.knowledge_graph.retriever.fusion_retriever import (
    KnowledgeGraphFusionRetriever,
)
from app.rag.indices.knowledge_graph.retriever.schema import RetrievedKnowledgeGraph
from app.rag.indices.vector_search.retriever.fusion_retriever import (
    VectorSearchFusionRetriever,
)
from app.rag.indices.vector_search.retriever.schema import RetrievedChunk
from app.exceptions import KBNotFound
from app.rag.llms.resolver import must_get_llm, get_llm
from .models import ChunksRetrivalRequest, KnowledgeGraphRetrivalRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/retrieve/chunks")
def retrieve_chunks(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    request: ChunksRetrivalRequest,
) -> List[RetrievedChunk]:
    try:
        config = request.retrieval_config
        llm = get_llm(db_session, config.llm_id)
        retriever = VectorSearchFusionRetriever(
            db_session=db_session,
            knowledge_base_ids=config.knowledge_base_ids,
            llm=llm,
            use_query_decompose=config.use_query_decompose,
            select_mode=config.kb_select_mode,
            config=config.vector_search,
        )
        return retriever.retrieve_chunks(QueryBundle(request.query))
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise


@router.get("/retrieve/knowledge_graph")
def retrieve_knowledge_graph(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    request: KnowledgeGraphRetrivalRequest,
) -> RetrievedKnowledgeGraph:
    try:
        config = request.retrieval_config
        llm = must_get_llm(db_session, config.llm_id)
        retriever = KnowledgeGraphFusionRetriever(
            db_session=db_session,
            knowledge_base_ids=config.knowledge_base_ids,
            llm=llm,
            use_query_decompose=config.use_query_decompose,
            select_mode=config.kb_select_mode,
            config=config.knowledge_graph,
        )
        return retriever.retrieve_knowledge_graph(QueryBundle(request.query))
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise
