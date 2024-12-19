import logging
from typing import List

from fastapi import Depends, APIRouter
from fastapi_pagination import Params, Page

from llama_index.core.schema import NodeWithScore, TextNode
from sqlalchemy import update

from app.api.admin_routes.llm.routes import LLMTestResult
from app.api.deps import CurrentSuperuserDep, SessionDep
from app.exceptions import RerankerModelNotFound, InternalServerError
from app.models import RerankerModel, AdminRerankerModel, ChatEngine
from app.rag.chat_config import get_reranker_model
from app.rag.reranker_model_option import (
    RerankerModelOption,
    admin_reranker_model_options,
)
from app.repositories.reranker_model import reranker_model_repo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/admin/reranker-models/options")
def get_reranker_model_options(user: CurrentSuperuserDep) -> List[RerankerModelOption]:
    return admin_reranker_model_options


@router.post("/admin/reranker-models/test")
def test_reranker_model(
    db_reranker_model: RerankerModel, user: CurrentSuperuserDep
) -> LLMTestResult:
    try:
        reranker = get_reranker_model(
            provider=db_reranker_model.provider,
            model=db_reranker_model.model,
            # for testing purpose, we only rerank 2 nodes
            top_n=2,
            config=db_reranker_model.config,
            credentials=db_reranker_model.credentials,
        )
        reranker.postprocess_nodes(
            nodes=[
                NodeWithScore(
                    node=TextNode(
                        text="TiDB is a distributed SQL database.",
                    ),
                    score=0.8,
                ),
                NodeWithScore(
                    node=TextNode(
                        text="TiDB is compatible with MySQL protocol.",
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
        success = True
        error = ""
    except Exception as e:
        success = False
        error = str(e)
    return LLMTestResult(success=success, error=error)


@router.get("/admin/reranker-models")
def list_reranker_models(
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> Page[AdminRerankerModel]:
    return reranker_model_repo.paginate(session, params)


@router.post("/admin/reranker-models")
def create_reranker_model(
    reranker_model: RerankerModel,
    session: SessionDep,
    user: CurrentSuperuserDep,
) -> AdminRerankerModel:
    return reranker_model_repo.create(session, reranker_model)


@router.get("/admin/reranker-models/{reranker_model_id}")
def get_reranker_model_detail(
    reranker_model_id: int,
    session: SessionDep,
    user: CurrentSuperuserDep,
) -> AdminRerankerModel:
    try:
        return reranker_model_repo.must_get(session, reranker_model_id)
    except RerankerModelNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.delete("/admin/reranker-models/{reranker_model_id}")
def delete_reranker_model(
    reranker_model_id: int,
    session: SessionDep,
    user: CurrentSuperuserDep,
):
    reranker_model = reranker_model_repo.must_get(session, reranker_model_id)

    # FIXME: Should be replaced with a new reranker or prohibit users from operating,
    #  If the current reranker is used by a Chat Engine or Knowledge Base.

    session.exec(
        update(ChatEngine)
        .where(ChatEngine.reranker_id == reranker_model_id)
        .values(reranker_id=None)
    )

    session.delete(reranker_model)
    session.commit()


@router.put("/admin/reranker-models/{reranker_model_id}/set_default")
def set_default_reranker_model(
    session: SessionDep, user: CurrentSuperuserDep, reranker_model_id: int
) -> AdminRerankerModel:
    try:
        reranker_model = reranker_model_repo.must_get(session, reranker_model_id)
        reranker_model_repo.set_default_model(session, reranker_model_id)
        session.refresh(reranker_model)
        return reranker_model
    except RerankerModelNotFound as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
