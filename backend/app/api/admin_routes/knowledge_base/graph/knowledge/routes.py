from fastapi import HTTPException
from starlette import status

from app.api.admin_routes.knowledge_base.graph.models import (
    KnowledgeRequest,
    KnowledgeNeighborRequest,
    KnowledgeChunkRequest,
)
from app.api.admin_routes.knowledge_base.graph.routes import router, logger
from app.api.deps import SessionDep
from app.exceptions import KBNotFound, InternalServerError
from app.rag.knowledge_base.index_store import get_kb_tidb_graph_store
from app.repositories import knowledge_base_repo


# Experimental interface


@router.post("/admin/knowledge_bases/{kb_id}/graph/knowledge")
def retrieve_knowledge(session: SessionDep, kb_id: int, request: KnowledgeRequest):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_store = get_kb_tidb_graph_store(session, kb)
        data = graph_store.retrieve_graph_data(
            request.query,
            request.top_k,
            request.similarity_threshold,
        )
        return {
            "entities": data["entities"],
            "relationships": data["relationships"],
        }
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/admin/knowledge_bases/{kb_id}/graph/knowledge/neighbors")
def retrieve_knowledge_neighbors(
    session: SessionDep, kb_id: int, request: KnowledgeNeighborRequest
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_store = get_kb_tidb_graph_store(session, kb)
        data = graph_store.retrieve_neighbors(
            request.entities_ids,
            request.query,
            request.max_depth,
            request.max_neighbors,
            request.similarity_threshold,
        )
        return data
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/admin/knowledge_bases/{kb_id}/graph/knowledge/chunks")
def retrieve_knowledge_chunks(
    session: SessionDep, kb_id: int, request: KnowledgeChunkRequest
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_store = get_kb_tidb_graph_store(session, kb)
        data = graph_store.get_chunks_by_relationships(request.relationships_ids)
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No chunks found for the given relationships",
            )
        return data
    except KBNotFound as e:
        raise e
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
