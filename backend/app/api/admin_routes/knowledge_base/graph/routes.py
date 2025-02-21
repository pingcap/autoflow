import logging

from fastapi import APIRouter, HTTPException
from app.api.admin_routes.knowledge_base.graph.models import (
    KBRetrieveKnowledgeGraphRequest,
    GraphSearchRequest,
    KnowledgeRequest,
)
from app.api.deps import SessionDep
from app.exceptions import InternalServerError
from app.rag.retrievers.knowledge_graph.schema import KnowledgeGraphRetrievalResult
from app.rag.knowledge_base.index_store import (
    get_kb_tidb_graph_store,
)
from app.rag.retrievers.knowledge_graph.simple_retriever import (
    KnowledgeGraphSimpleRetriever,
)
from app.repositories import knowledge_base_repo

router = APIRouter(
    prefix="/admin/knowledge_bases/{kb_id}/graph",
    tags=["knowledge_base/graph"],
)
logger = logging.getLogger(__name__)


@router.post("/retrieve")
def retrieve_knowledge_graph(
    db_session: SessionDep, kb_id: int, request: KBRetrieveKnowledgeGraphRequest
) -> KnowledgeGraphRetrievalResult:
    try:
        retriever = KnowledgeGraphSimpleRetriever(
            db_session=db_session,
            knowledge_base_id=kb_id,
            config=request.retrival_config.knowledge_graph,
        )
        knowledge_graph = retriever.retrieve_knowledge_graph(request.query)
        return KnowledgeGraphRetrievalResult(
            entities=knowledge_graph.entities,
            relationships=knowledge_graph.relationships,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/knowledge", deprecated=True)
def retrieve_knowledge(session: SessionDep, kb_id: int, request: KnowledgeRequest):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_store = get_kb_tidb_graph_store(session, kb)
        return graph_store.retrieve_subgraph_by_similar(
            request.query,
            request.top_k,
            request.similarity_threshold,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


# Legacy


@router.post("/search", deprecated=True)
def legacy_search_graph(session: SessionDep, kb_id: int, request: GraphSearchRequest):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_store = get_kb_tidb_graph_store(session, kb)
        entities, relationships = graph_store.retrieve_with_weight(
            request.query,
            [],
            request.depth,
            request.include_meta,
            request.with_degree,
            request.relationship_meta_filters,
        )
        return {
            "entities": entities,
            "relationships": relationships,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
