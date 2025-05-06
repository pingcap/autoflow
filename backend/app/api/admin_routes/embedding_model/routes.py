from typing import List

from fastapi import APIRouter, Depends
from fastapi_pagination import Params, Page

from app.api.admin_routes.embedding_model.models import (
    EmbeddingModelItem,
    EmbeddingModelDetail,
    EmbeddingModelUpdate,
    EmbeddingModelTestResult,
    EmbeddingModelCreate,
)
from app.api.deps import CurrentSuperuserDep, SessionDep
from app.repositories.embedding_model import embedding_model_repo
from app.rag.embeddings.provider import (
    EmbeddingProviderOption,
    embedding_provider_options,
)
from app.rag.embeddings.resolver import resolve_embed_model
from app.logger import logger

router = APIRouter()


@router.get("/admin/embedding-models/providers/options")
def list_embedding_model_provider_options(
    user: CurrentSuperuserDep,
) -> List[EmbeddingProviderOption]:
    return embedding_provider_options


@router.get("/admin/embedding-models")
def list_embedding_models(
    db_session: SessionDep, user: CurrentSuperuserDep, params: Params = Depends()
) -> Page[EmbeddingModelItem]:
    return embedding_model_repo.paginate(db_session, params)


@router.post("/admin/embedding-models/test")
def test_embedding_model(
    user: CurrentSuperuserDep,
    create: EmbeddingModelCreate,
) -> EmbeddingModelTestResult:
    try:
        embed_model = resolve_embed_model(
            provider=create.provider,
            model=create.model,
            config=create.config,
            credentials=create.credentials,
        )
        embedding = embed_model.get_query_embedding("Hello, world!")
        expected_length = create.vector_dimension
        if len(embedding) != expected_length:
            raise ValueError(
                f"Embedding model is configured with {expected_length} dimensions, but got vector embedding with {len(embedding)} dimensions."
            )
        success = True
        error = ""
    except Exception as e:
        logger.info(f"Failed to test embedding model: {e}")
        success = False
        error = str(e)
    return EmbeddingModelTestResult(success=success, error=error)


@router.post("/admin/embedding-models")
def create_embedding_model(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    create: EmbeddingModelCreate,
) -> EmbeddingModelDetail:
    return embedding_model_repo.create(db_session, create)


@router.get("/admin/embedding-models/{model_id}")
def get_embedding_model_detail(
    db_session: SessionDep, user: CurrentSuperuserDep, model_id: int
) -> EmbeddingModelDetail:
    return embedding_model_repo.must_get(db_session, model_id)


@router.put("/admin/embedding-models/{model_id}")
def update_embedding_model(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    model_id: int,
    update: EmbeddingModelUpdate,
) -> EmbeddingModelDetail:
    embed_model = embedding_model_repo.must_get(db_session, model_id)
    return embedding_model_repo.update(db_session, embed_model, update)


@router.delete("/admin/embedding-models/{model_id}")
def delete_embedding_model(
    db_session: SessionDep, user: CurrentSuperuserDep, model_id: int
) -> None:
    embedding_model = embedding_model_repo.must_get(db_session, model_id)
    embedding_model_repo.delete(db_session, embedding_model)


@router.put("/admin/embedding-models/{model_id}/set_default")
def set_default_embedding_model(
    db_session: SessionDep, user: CurrentSuperuserDep, model_id: int
) -> EmbeddingModelDetail:
    embed_model = embedding_model_repo.must_get(db_session, model_id)
    return embedding_model_repo.set_default(db_session, embed_model)
