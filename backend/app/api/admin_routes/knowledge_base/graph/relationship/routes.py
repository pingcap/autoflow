import logging

from typing import Annotated, List
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi_pagination import Params, Page

from app.api.admin_routes.knowledge_base.graph.models import (
    RelationshipUpdate,
    RelationshipBatchRequest,
)
from app.api.deps import SessionDep
from app.exceptions import InternalServerError
from app.models import RelationshipPublic, Chunk as DBChunk
from app.rag.indices.knowledge_graph.schema import (
    RelationshipCreate,
    RelationshipFilters,
)
from app.rag.knowledge_base.index_store import get_kb_graph_editor
from app.repositories import knowledge_base_repo

router = APIRouter(
    prefix="/admin/knowledge_bases/{kb_id}/graph/relationships",
    tags=["knowledge_base/graph/relationship"],
)
logger = logging.getLogger(__name__)


@router.get("/", response_model=Page[RelationshipPublic])
def query_relationships(
    db_session: SessionDep,
    kb_id: int,
    filters: Annotated[RelationshipFilters, Query()] = RelationshipFilters(),
    params: Params = Depends(),
):
    try:
        kb = knowledge_base_repo.must_get(db_session, kb_id)
        graph_editor = get_kb_graph_editor(db_session, kb)
        return graph_editor.query_relationships(filters, params)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/", response_model=RelationshipPublic)
def create_relationship(
    session: SessionDep,
    kb_id: int,
    create: RelationshipCreate,
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        return graph_editor.create_relationship(create)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/chunks", response_model=List[DBChunk])
def batch_get_chunks_by_relationships(
    session: SessionDep,
    kb_id: int,
    request: RelationshipBatchRequest,
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        return graph_editor.batch_get_chunks_by_relationships(request.relationship_ids)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/{relationship_id}", response_model=RelationshipPublic)
def get_relationship(session: SessionDep, kb_id: int, relationship_id: int):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        return graph_editor.must_get_relationship(relationship_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.put("/{relationship_id}", response_model=RelationshipPublic)
def update_relationship(
    session: SessionDep,
    kb_id: int,
    relationship_id: int,
    update: RelationshipUpdate,
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        return graph_editor.update_relationship(relationship_id, update)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.delete("/{relationship_id}")
def delete_relationship(session: SessionDep, kb_id: int, relationship_id: int):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        return graph_editor.delete_relationship(relationship_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
