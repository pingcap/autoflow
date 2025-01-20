import logging
from typing import List

from fastapi import APIRouter
from llama_index.core import QueryBundle
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.indices.knowledge_graph.retriever.fusion_retriever import (
    KnowledgeGraphFusionRetriever,
)
from app.rag.indices.knowledge_graph.schema import RetrievedKnowledgeGraph
from app.rag.indices.vector_search.retriever.fusion_retriever import (
    VectorSearchFusionRetriever,
)
from app.rag.indices.vector_search.schema import RetrievedChunk
from app.exceptions import KBNotFound
from app.rag.llms.resolver import must_get_llm
from app.utils.dspy import get_dspy_lm_by_llama_llm
from .models import RetrieveChunksRequest, RetrieveKnowledgeGraphRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/admin/retrieve/chunks")
def retrieve_chunks(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    request: RetrieveChunksRequest,
) -> List[RetrievedChunk]:
    try:
        vector_search_config = request.retrieval_config.vector_search
        retriever = VectorSearchFusionRetriever(
            db_session=db_session,
            knowledge_base_id=request.knowledge_base_ids[0],
            config=vector_search_config,
        )
        return retriever.retrieve_chunks(QueryBundle(request.query), db_session)
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise


@router.get("/admin/retrieve/knowledge_graph")
def retrieve_knowledge_graph(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    request: RetrieveKnowledgeGraphRequest,
) -> RetrievedKnowledgeGraph:
    try:
        llm = must_get_llm(db_session, request.knowledge_base_id)
        dspy_lm = get_dspy_lm_by_llama_llm(llm)
        knowledge_graph_config = request.retrival_config.knowledge_graph
        retriever = KnowledgeGraphFusionRetriever(
            llm=llm,
            dspy_lm=dspy_lm,
            knowledge_base_ids=request.knowledge_base_ids,
            db_session=db_session,
            config=knowledge_graph_config,
        )
        entities, relationships = retriever.retrieve_knowledge_graph(
            query_bundle=QueryBundle(request.query)
        )
        return RetrievedKnowledgeGraph(
            entities=entities,
            relationships=relationships,
        )
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise
