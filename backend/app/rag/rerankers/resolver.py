from typing import Optional
from sqlmodel import Session

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.xinference_rerank import XinferenceRerank
from llama_index.postprocessor.bedrock_rerank import AWSBedrockRerank

from app.rag.rerankers.baisheng.baisheng_reranker import BaishengRerank
from app.rag.rerankers.local.local_reranker import LocalRerank
from app.rag.rerankers.vllm.vllm_reranker import VLLMRerank
from app.rag.rerankers.provider import RerankerProvider

from app.repositories.reranker_model import reranker_model_repo


def get_reranker_model(
    provider: RerankerProvider,
    model: str,
    top_n: int,
    config: dict,
    credentials: str | list | dict | None,
) -> BaseNodePostprocessor:
    match provider:
        case RerankerProvider.JINA:
            return JinaRerank(
                model=model,
                top_n=top_n,
                api_key=credentials,
                **config,
            )
        case RerankerProvider.COHERE:
            return CohereRerank(
                model=model,
                top_n=top_n,
                api_key=credentials,
                **config,
            )
        case RerankerProvider.BAISHENG:
            return BaishengRerank(
                model=model,
                top_n=top_n,
                api_key=credentials,
                **config,
            )
        case RerankerProvider.LOCAL:
            return LocalRerank(
                model=model,
                top_n=top_n,
                **config,
            )
        case RerankerProvider.VLLM:
            return VLLMRerank(
                model=model,
                top_n=top_n,
                **config,
            )
        case RerankerProvider.XINFERENCE:
            return XinferenceRerank(
                model=model,
                top_n=top_n,
                **config,
            )
        case RerankerProvider.BEDROCK:
            return AWSBedrockRerank(
                rerank_model_name=model,
                top_n=top_n,
                aws_access_key_id=credentials["aws_access_key_id"],
                aws_secret_access_key=credentials["aws_secret_access_key"],
                region_name=credentials["aws_region_name"],
                **config,
            )
        case _:
            raise ValueError(f"Got unknown reranker provider: {provider}")


# FIXME: Reranker top_n should be config in the retrival config.
def get_default_reranker_model(
    session: Session, top_n: int = None
) -> Optional[BaseNodePostprocessor]:
    db_reranker = reranker_model_repo.get_default(session)
    if not db_reranker:
        return None
    top_n = db_reranker.top_n if top_n is None else top_n
    return get_reranker_model(
        db_reranker.provider,
        db_reranker.model,
        top_n,
        db_reranker.config,
        db_reranker.credentials,
    )


def must_get_default_reranker_model(session: Session) -> BaseNodePostprocessor:
    db_reranker = reranker_model_repo.must_get_default(session)
    return get_reranker_model(
        db_reranker.provider,
        db_reranker.model,
        db_reranker.top_n,
        db_reranker.config,
        db_reranker.credentials,
    )
