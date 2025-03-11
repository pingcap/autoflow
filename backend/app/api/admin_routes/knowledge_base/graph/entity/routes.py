import logging

from typing import List, Annotated
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi_pagination import Params, Page

from app.api.deps import SessionDep
from app.exceptions import InternalServerError
from app.models import EntityPublic, EntityType
from app.rag.indices.knowledge_graph.schema import (
    EntityCreate,
    EntityFilters,
    SynopsisEntityCreate,
    EntityUpdate,
)
from app.rag.knowledge_base.index_store import (
    get_kb_graph_editor,
    get_kb_tidb_graph_store,
)
from app.rag.retrievers.knowledge_graph.schema import (
    RetrievedEntity,
    RetrievedKnowledgeGraph,
)
from app.repositories import knowledge_base_repo

router = APIRouter(
    prefix="/admin/knowledge_bases/{kb_id}/graph/entities",
    tags=["knowledge_base/graph/entity"],
)
logger = logging.getLogger(__name__)


@router.get("/", response_model=Page[EntityPublic])
def list_entities(
    db_session: SessionDep,
    kb_id: int,
    filters: Annotated[EntityFilters, Query()] = EntityFilters(),
    params: Params = Depends(),
):
    try:
        kb = knowledge_base_repo.must_get(db_session, kb_id)
        graph_editor = get_kb_graph_editor(db_session, kb)
        return graph_editor.query_entities(filters, params)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/", response_model=EntityPublic)
def create_entity(session: SessionDep, kb_id: int, create: EntityCreate):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        return graph_editor.create_entity(create)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/synopsis", response_model=EntityPublic)
def create_synopsis_entity(
    session: SessionDep, kb_id: int, create: SynopsisEntityCreate
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        return graph_editor.create_synopsis_entity(create)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get(
    "/search",
)
def search_similar_entities(
    session: SessionDep,
    kb_id: int,
    query: str,
    top_k: int = 10,
    nprobe: int = 10,
    entity_type: EntityType = EntityType.original,
    similarity_threshold: float = 0.4,
) -> List[RetrievedEntity]:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_store = get_kb_tidb_graph_store(session, kb)
        return graph_store.retrieve_entities(
            query=query,
            top_k=top_k,
            nprobe=nprobe,
            entity_type=entity_type,
            similarity_threshold=similarity_threshold,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/{entity_id}", response_model=EntityPublic)
def get_entity(session: SessionDep, kb_id: int, entity_id: int):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        return graph_editor.must_get_entity(entity_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.put("/{entity_id}", response_model=EntityPublic)
def update_entity(
    session: SessionDep, kb_id: int, entity_id: int, update: EntityUpdate
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        return graph_editor.update_entity(entity_id, update)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.delete("/{entity_id}")
def delete_entity(session: SessionDep, kb_id: int, entity_id: int):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        graph_editor.delete_entity(entity_id)
        return {
            "detail": "success",
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/{entity_id}/subgraph")
def get_entity_subgraph(
    session: SessionDep, kb_id: int, entity_id: int
) -> RetrievedKnowledgeGraph:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_editor = get_kb_graph_editor(session, kb)
        return graph_editor.get_entity_subgraph(entity_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
