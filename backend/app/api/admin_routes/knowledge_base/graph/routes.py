import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.api.admin_routes.knowledge_base.graph.models import (
    SynopsisEntityCreate,
    EntityUpdate,
    RelationshipUpdate,
    GraphSearchRequest,
    KnowledgeRequest,
    KnowledgeNeighborRequest,
    KnowledgeChunkRequest,
)
from app.api.deps import SessionDep
from app.exceptions import KBNotFound, InternalServerError
from app.models import (
    EntityPublic,
    RelationshipPublic,
)
from app.rag.knowledge_base.index_store import (
    get_kb_tidb_graph_editor,
    get_kb_tidb_graph_store,
)
from app.repositories import knowledge_base_repo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/admin/knowledge_bases/{kb_id}/graph/entities/search",
    response_model=List[EntityPublic],
)
def search_similar_entities(
    session: SessionDep, kb_id: int, query: str, top_k: int = 10
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        tidb_graph_editor = get_kb_tidb_graph_editor(session, kb)
        return tidb_graph_editor.search_similar_entities(session, query, top_k)
    except KBNotFound as e:
        raise e
    except Exception as e:
        # TODO: throw InternalServerError
        raise e


@router.post(
    "/admin/knowledge_bases/{kb_id}/graph/entities/synopsis",
    response_model=EntityPublic,
)
def create_synopsis_entity(
    session: SessionDep, kb_id: int, request: SynopsisEntityCreate
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        tidb_graph_editor = get_kb_tidb_graph_editor(session, kb)
        return tidb_graph_editor.create_synopsis_entity(
            session,
            request.name,
            request.description,
            request.topic,
            request.meta,
            request.entities,
        )
    except KBNotFound as e:
        raise e
    except Exception as e:
        # TODO: throw InternalServerError
        raise e


@router.get(
    "/admin/knowledge_bases/{kb_id}/graph/entities/{entity_id}",
    response_model=EntityPublic,
)
def get_entity(session: SessionDep, kb_id: int, entity_id: int):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        tidb_graph_editor = get_kb_tidb_graph_editor(session, kb)
        entity = tidb_graph_editor.get_entity(session, entity_id)
        if not entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Entity not found",
            )
        return entity
    except KBNotFound as e:
        raise e
    except Exception as e:
        # TODO: throw InternalServerError
        raise e


@router.put(
    "/admin/knowledge_bases/{kb_id}/graph/entities/{entity_id}",
    response_model=EntityPublic,
)
def update_entity(
    session: SessionDep, kb_id: int, entity_id: int, entity_update: EntityUpdate
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        tidb_graph_editor = get_kb_tidb_graph_editor(session, kb)
        old_entity = tidb_graph_editor.get_entity(session, entity_id)
        if old_entity is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Entity not found",
            )
        entity = tidb_graph_editor.update_entity(
            session, old_entity, entity_update.model_dump()
        )
        return entity
    except KBNotFound as e:
        raise e
    except Exception as e:
        # TODO: throw InternalServerError
        raise e


@router.get("/admin/knowledge_bases/{kb_id}/graph/entities/{entity_id}/subgraph")
def get_entity_subgraph(session: SessionDep, kb_id: int, entity_id: int) -> dict:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        tidb_graph_editor = get_kb_tidb_graph_editor(session, kb)
        entity = tidb_graph_editor.get_entity(session, entity_id)
        if entity is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Entity not found",
            )
        relationships, entities = tidb_graph_editor.get_entity_subgraph(session, entity)
        return {
            "relationships": relationships,
            "entities": entities,
        }
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get(
    "/admin/knowledge_bases/{kb_id}/graph/relationships/{relationship_id}",
    response_model=RelationshipPublic,
)
def get_relationship(session: SessionDep, kb_id: int, relationship_id: int):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        tidb_graph_editor = get_kb_tidb_graph_editor(session, kb)
        relationship = tidb_graph_editor.get_relationship(session, relationship_id)
        if relationship is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Relationship not found",
            )
        return relationship
    except KBNotFound as e:
        raise e
    except Exception as e:
        # TODO: throw InternalServerError
        raise e


@router.put(
    "/admin/knowledge_bases/{kb_id}/graph/relationships/{relationship_id}",
    response_model=RelationshipPublic,
)
def update_relationship(
    session: SessionDep,
    kb_id: int,
    relationship_id: int,
    relationship_update: RelationshipUpdate,
):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        tidb_graph_editor = get_kb_tidb_graph_editor(session, kb)
        old_relationship = tidb_graph_editor.get_relationship(session, relationship_id)
        if old_relationship is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Relationship not found",
            )
        relationship = tidb_graph_editor.update_relationship(
            session, old_relationship, relationship_update.model_dump()
        )
        return relationship
    except KBNotFound as e:
        raise e
    except Exception as e:
        # TODO: throw InternalServerError
        raise e


@router.post("/admin/knowledge_bases/{kb_id}/graph/search")
def search_graph(session: SessionDep, kb_id: int, request: GraphSearchRequest):
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        graph_store = get_kb_tidb_graph_store(session, kb)
        entities, relations, _ = graph_store.retrieve_with_weight(
            request.query,
            [],
            request.depth,
            request.include_meta,
            request.with_degree,
            False,
            {},
        )
        return {
            "entities": entities,
            "relationships": relations,
        }
    except KBNotFound as e:
        raise e
    except Exception as e:
        # TODO: throw InternalServerError
        raise e


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
