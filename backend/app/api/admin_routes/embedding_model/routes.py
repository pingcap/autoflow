import logging
from typing import List

from fastapi import APIRouter, Depends
from fastapi_pagination import Params, Page

from app.api.admin_routes.embedding_model.models import (
    EmbeddingModelItem,
    EmbeddingModelDetail,
    EmbeddingModelUpdate,
    EmbeddingModelTestResult, EmbeddingModelCreate
)
from app.api.deps import CurrentSuperuserDep, SessionDep
from app.exceptions import EmbeddingModelNotFoundError, InternalServerError
from app.rag.chat_config import get_embedding_model
from app.rag.embed_model_option import EmbeddingModelOption, admin_embed_model_options
from app.repositories.embedding_model import embedding_model_repo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/admin/embedding-models/options")
def get_embedding_model_options(
    user: CurrentSuperuserDep,
) -> List[EmbeddingModelOption]:
    return admin_embed_model_options


@router.post("/admin/embedding-models")
def create_embedding_model(
    session: SessionDep,
    user: CurrentSuperuserDep,
    create: EmbeddingModelCreate,
) -> EmbeddingModelDetail:
    try:
        return embedding_model_repo.create(session, create)
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/admin/embedding-models/test")
def test_embedding_model(
    user: CurrentSuperuserDep,
    create: EmbeddingModelCreate,
) -> EmbeddingModelTestResult:
    try:
        embed_model = get_embedding_model(
            provider=create.provider,
            model=create.model,
            config=create.config,
            credentials=create.credentials,
        )
        embedding = embed_model.get_query_embedding("Hello, world!")
        expected_length = create.vector_dimension
        if len(embedding) != expected_length:
            raise ValueError(
                f"Currently we only support {expected_length} dims embedding, got {len(embedding)} dims."
            )
        success = True
        error = ""
    except Exception as e:
        success = False
        error = str(e)
    return EmbeddingModelTestResult(success=success, error=error)


@router.get("/admin/embedding-models")
def list_embedding_models(
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends()
) -> Page[EmbeddingModelItem]:
    return embedding_model_repo.paginate(session, params)


@router.get("/admin/embedding-models/{model_id}")
def get_embedding_model_detail(
    session: SessionDep,
    user: CurrentSuperuserDep,
    model_id: int
) -> EmbeddingModelDetail:
    try:
        return embedding_model_repo.must_get(session, model_id)
    except EmbeddingModelNotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.put("/admin/embedding-models/{model_id}")
def update_embedding_model(
    session: SessionDep,
    user: CurrentSuperuserDep,
    model_id: int,
    update: EmbeddingModelUpdate,
) -> EmbeddingModelDetail:
    try:
        embed_model = embedding_model_repo.must_get(session, model_id)
        embedding_model_repo.update(session, embed_model, update)
        return embed_model
    except EmbeddingModelNotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.put("/admin/embedding-models/{model_id}/set_default")
def set_default_embedding_model(
    session: SessionDep,
    user: CurrentSuperuserDep,
    model_id: int
) -> EmbeddingModelDetail:
    try:
        embed_model = embedding_model_repo.must_get(session, model_id)
        embedding_model_repo.set_default_model(session, model_id)
        session.refresh(embed_model)
        return embed_model
    except EmbeddingModelNotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()