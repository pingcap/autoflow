import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.api.admin_routes.knowledge_base.graph.models import (
    SynopsisEntityCreate,
    EntityUpdate,
    RelationshipUpdate,
    KBRetrieveKnowledgeGraphRequest,
    GraphSearchRequest,
)
from app.api.deps import SessionDep
from app.exceptions import KBNotFound, InternalServerError
from app.models import (
    EntityPublic,
    RelationshipPublic,
)
from app.rag.retrievers.knowledge_graph.schema import (
    KnowledgeGraphRetrievalResult,
)
from app.rag.knowledge.base import KnowledgeBase
from app.rag.retrievers.knowledge_graph.simple_retriever import (
    KnowledgeGraphSimpleRetriever,
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
        kb_model = knowledge_base_repo.must_get(session, kb_id)
        kb = KnowledgeBase.load_from_db(session, kb_model)
        return kb.graph_editor.search_similar_entities(session, query, top_k)
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
        kb_model = knowledge_base_repo.must_get(session, kb_id)
        kb = KnowledgeBase.load_from_db(session, kb_model)
        return kb.graph_editor.create_synopsis_entity(
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
        kb_model = knowledge_base_repo.must_get(session, kb_id)
        kb = KnowledgeBase.load_from_db(session, kb_model)
        entity = kb.graph_editor.get_entity(session, entity_id)
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
        kb_model = knowledge_base_repo.must_get(session, kb_id)
        kb = KnowledgeBase.load_from_db(session, kb_model)
        old_entity = kb.graph_editor.get_entity(session, entity_id)
        if old_entity is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Entity not found",
            )
        entity = kb.graph_editor.update_entity(
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
        kb_model = knowledge_base_repo.must_get(session, kb_id)
        kb = KnowledgeBase.load_from_db(session, kb_model)
        entity = kb.graph_editor.get_entity(session, entity_id)
        if entity is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Entity not found",
            )
        relationships, entities = kb.graph_editor.get_entity_subgraph(session, entity)
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
        kb_model = knowledge_base_repo.must_get(session, kb_id)
        kb = KnowledgeBase.load_from_db(session, kb_model)
        relationship = kb.graph_editor.get_relationship(session, relationship_id)
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
        kb_model = knowledge_base_repo.must_get(session, kb_id)
        kb = KnowledgeBase.load_from_db(session, kb_model)
        old_relationship = kb.graph_editor.get_relationship(session, relationship_id)
        if old_relationship is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Relationship not found",
            )
        relationship = kb.graph_editor.update_relationship(
            session, old_relationship, relationship_update.model_dump()
        )
        return relationship
    except KBNotFound as e:
        raise e
    except Exception as e:
        # TODO: throw InternalServerError
        raise e


@router.post("/admin/knowledge_bases/{kb_id}/graph/retrieve")
def retrieve_kb_knowledge_graph(
    db_session: SessionDep, kb_id: int, request: KBRetrieveKnowledgeGraphRequest
) -> KnowledgeGraphRetrievalResult:
    try:
        kb_model = knowledge_base_repo.must_get(db_session, kb_id)
        kb = KnowledgeBase.load_from_db(db_session, kb_model)
        retriever = KnowledgeGraphSimpleRetriever(kb.graph_store)
        return retriever.retrieve(request.query, request.top_k)
    except KBNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/admin/knowledge_bases/{kb_id}/graph/search", deprecated=True)
def legacy_search_graph(session: SessionDep, kb_id: int, request: GraphSearchRequest):
    try:
        kb_model = knowledge_base_repo.must_get(session, kb_id)
        kb = KnowledgeBase.load_from_db(session, kb_model)
        data = kb.graph_store.retrieve_graph_data(
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
