from typing import List

from fastapi import Depends, APIRouter
from fastapi_pagination import Params, Page
from pydantic import BaseModel
from llama_index.core.schema import NodeWithScore, TextNode

from app.api.admin_routes.llm.routes import LLMTestResult
from app.api.deps import CurrentSuperuserDep, SessionDep
from app.models import RerankerModel, AdminRerankerModel
from app.models.reranker_model import RerankerModelUpdate
from app.repositories.reranker_model import reranker_model_repo
from app.rag.rerankers.provider import RerankerProviderOption, reranker_provider_options
from app.rag.rerankers.resolver import resolve_reranker

from app.logger import logger


router = APIRouter()


@router.get("/admin/reranker-models/providers/options")
def list_reranker_model_provider_options(
    user: CurrentSuperuserDep,
) -> List[RerankerProviderOption]:
    return reranker_provider_options


@router.get("/admin/reranker-models")
def list_reranker_models(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> Page[AdminRerankerModel]:
    return reranker_model_repo.paginate(db_session, params)


class RerankerModelTestResult(BaseModel):
    success: bool
    error: str = ""


@router.post("/admin/reranker-models/test")
def test_reranker_model(
    db_reranker_model: RerankerModel, user: CurrentSuperuserDep
) -> LLMTestResult:
    try:
        reranker = resolve_reranker(
            provider=db_reranker_model.provider,
            model=db_reranker_model.model,
            # for testing purpose, we only rerank 2 nodes
            top_n=2,
            config=db_reranker_model.config,
            credentials=db_reranker_model.credentials,
        )
        reranked_nodes = reranker.postprocess_nodes(
            nodes=[
                NodeWithScore(
                    node=TextNode(
                        text="TiDB is a distributed SQL database.",
                    ),
                    score=0.8,
                ),
                NodeWithScore(
                    node=TextNode(
                        text="TiKV is a distributed key-value storage engine.",
                    ),
                    score=0.6,
                ),
                NodeWithScore(
                    node=TextNode(
                        text="TiFlash is a columnar storage engine.",
                    ),
                    score=0.4,
                ),
            ],
            query_str="What is TiDB?",
        )
        if len(reranked_nodes) != 2:
            raise ValueError("expected 2 nodes, but got %d", len(reranked_nodes))
        success = True
        error = ""
    except Exception as e:
        logger.info(f"Failed to test reranker model: {e}")
        success = False
        error = str(e)
    return RerankerModelTestResult(success=success, error=error)


@router.post("/admin/reranker-models")
def create_reranker_model(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    reranker_model: RerankerModel,
) -> AdminRerankerModel:
    return reranker_model_repo.create(db_session, reranker_model)


@router.get("/admin/reranker-models/{model_id}")
def get_reranker_model(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    model_id: int,
) -> AdminRerankerModel:
    return reranker_model_repo.must_get(db_session, model_id)


@router.put("/admin/reranker-models/{model_id}")
def update_reranker_model(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    model_id: int,
    model_update: RerankerModelUpdate,
) -> AdminRerankerModel:
    reranker_model = reranker_model_repo.must_get(db_session, model_id)
    return reranker_model_repo.update(db_session, reranker_model, model_update)


@router.delete("/admin/reranker-models/{model_id}")
def delete_reranker_model(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    model_id: int,
) -> None:
    reranker_model = reranker_model_repo.must_get(db_session, model_id)
    reranker_model_repo.delete(db_session, reranker_model)


@router.put("/admin/reranker-models/{model_id}/set_default")
def set_default_reranker_model(
    db_session: SessionDep, user: CurrentSuperuserDep, model_id: int
) -> AdminRerankerModel:
    reranker_model = reranker_model_repo.must_get(db_session, model_id)
    return reranker_model_repo.set_default(db_session, reranker_model)
