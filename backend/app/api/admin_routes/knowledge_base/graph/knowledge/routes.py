from fastapi import HTTPException
from pydantic import BaseModel
from starlette import status

from app.api.admin_routes.knowledge_base.graph.models import (
    KnowledgeRequest,
    KnowledgeNeighborRequest,
)
from app.api.admin_routes.knowledge_base.graph.routes import router, logger
from app.api.deps import SessionDep
from app.exceptions import KBNotFound, InternalServerError
from app.rag.knowledge_base.index_store import (
    get_kb_tidb_graph_store,
    get_kb_graph_editor,
)
from app.repositories import knowledge_base_repo


# Experimental interface


@router.post("/knowledge", deprecated=True)
def legacy_retrieve_knowledge(
    session: SessionDep, kb_id: int, request: KnowledgeRequest
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_store = get_kb_tidb_graph_store(session, kb)
        data = graph_store.retrieve_subgraph_by_similar(
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


@router.post("/knowledge/neighbors", deprecated=True)
def legacy_retrieve_knowledge_neighbors(
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


class KnowledgeChunkRequest(BaseModel):
    pass


@router.post("/knowledge/chunks", deprecated=True)
def legacy_retrieve_knowledge_chunks(
    session: SessionDep, kb_id: int, request: KnowledgeChunkRequest
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        data = graph_editor.batch_get_chunks_by_relationships(request.relationships_ids)
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No chunks found for the given relationships",
            )
        return data
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
