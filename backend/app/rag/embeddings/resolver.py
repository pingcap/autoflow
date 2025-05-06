from typing import Optional

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from sqlmodel import Session

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from app.rag.embeddings.open_like.openai_like_embedding import OpenAILikeEmbedding
from app.rag.embeddings.local.local_embedding import LocalEmbedding

from app.repositories.embedding_model import embedding_model_repo
from app.rag.embeddings.provider import EmbeddingProvider


def resolve_embed_model(
    provider: EmbeddingProvider,
    model: str,
    config: dict,
    credentials: str | list | dict | None,
) -> BaseEmbedding:
    match provider:
        case EmbeddingProvider.OPENAI:
            return OpenAIEmbedding(
                model=model,
                api_key=credentials,
                **config,
            )
        case EmbeddingProvider.JINA:
            return JinaEmbedding(
                model=model,
                api_key=credentials,
                **config,
            )
        case EmbeddingProvider.COHERE:
            return CohereEmbedding(
                model_name=model,
                cohere_api_key=credentials,
                **config,
            )
        case EmbeddingProvider.BEDROCK:
            return BedrockEmbedding(
                model_name=model,
                aws_access_key_id=credentials["aws_access_key_id"],
                aws_secret_access_key=credentials["aws_secret_access_key"],
                region_name=credentials["aws_region_name"],
                **config,
            )
        case EmbeddingProvider.OLLAMA:
            return OllamaEmbedding(
                model_name=model,
                **config,
            )
        case EmbeddingProvider.LOCAL:
            return LocalEmbedding(
                model=model,
                **config,
            )
        case EmbeddingProvider.GITEEAI:
            return OpenAILikeEmbedding(
                model=model,
                api_base="https://ai.gitee.com/v1",
                api_key=credentials,
                **config,
            )
        case EmbeddingProvider.AZURE_OPENAI:
            return AzureOpenAIEmbedding(
                model=model,
                api_key=credentials,
                **config,
            )
        case EmbeddingProvider.OPENAI_LIKE:
            return OpenAILikeEmbedding(
                model=model,
                api_key=credentials,
                **config,
            )
        case _:
            raise ValueError(f"Got unknown embedding provider: {provider}")


def get_default_embed_model(session: Session) -> Optional[BaseEmbedding]:
    db_embed_model = embedding_model_repo.get_default(session)
    if not db_embed_model:
        return None
    return resolve_embed_model(
        db_embed_model.provider,
        db_embed_model.model,
        db_embed_model.config,
        db_embed_model.credentials,
    )


def must_get_default_embed_model(session: Session) -> BaseEmbedding:
    db_embed_model = embedding_model_repo.must_get_default(session)
    return resolve_embed_model(
        db_embed_model.provider,
        db_embed_model.model,
        db_embed_model.config,
        db_embed_model.credentials,
    )
