from app.security.provider_http import openai_http_clients, ProviderSession
from app.security.provider_config import prepare_provider_config
from typing import Optional

from llama_index.core.llms.llm import LLM
from sqlmodel import Session

from app.repositories.llm import llm_repo
from app.rag.llms.provider import LLMProvider


def resolve_llm(
    provider: LLMProvider,
    model: str,
    config: Optional[dict] = {},
    credentials: Optional[str | list | dict] = None,
) -> LLM:
    config = prepare_provider_config("llm", provider, model, config, credentials)
    match provider:
        case LLMProvider.OPENAI:
            from llama_index.llms.openai import OpenAI

            return OpenAI(
                **openai_http_clients(),
                model=model,
                api_key=credentials,
                **config,
            )
        case LLMProvider.OPENAI_LIKE:
            from llama_index.llms.openai_like import OpenAILike

            config.setdefault("is_chat_model", True)
            config.setdefault("context_window", 200 * 1000)
            return OpenAILike(
                model=model, api_key=credentials, **openai_http_clients(), **config
            )
        case LLMProvider.BEDROCK:
            from llama_index.llms.bedrock_converse import BedrockConverse

            access_key_id = credentials["aws_access_key_id"]
            secret_access_key = credentials["aws_secret_access_key"]
            region_name = credentials["aws_region_name"]

            config.setdefault("max_tokens", 4096)

            return BedrockConverse(
                model=model,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name,
                **config,
            )
        case LLMProvider.GEMINI:
            from llama_index.llms.google_genai import GoogleGenAI

            from google.genai.types import HttpOptions

            return GoogleGenAI(
                model=model,
                api_key=credentials,
                http_options=HttpOptions(
                    timeout=int(config.pop("timeout", 60) * 1000),
                    client_args={"trust_env": False, "follow_redirects": False},
                    async_client_args={
                        "trust_env": False,
                        "follow_redirects": False,
                        "allow_redirects": False,
                    },
                ),
                **config,
            )
        case LLMProvider.VERTEX_AI | LLMProvider.ANTHROPIC_VERTEX:
            from llama_index.llms.google_genai import GoogleGenAI
            from llama_index.llms.google_genai.base import VertexAIConfig
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request

            google_creds: service_account.Credentials = (
                service_account.Credentials.from_service_account_info(
                    credentials,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            )
            with ProviderSession(
                "https://oauth2.googleapis.com/token"
            ) as oauth_session:
                google_creds.refresh(request=Request(session=oauth_session))
            project = credentials.get("project_id") or config.get("project")
            location = config.get("location", "us-west1")

            from google.genai.types import HttpOptions

            llm = GoogleGenAI(
                http_options=HttpOptions(
                    timeout=int(config.pop("timeout", 60) * 1000),
                    client_args={"trust_env": False, "follow_redirects": False},
                    async_client_args={
                        "trust_env": False,
                        "follow_redirects": False,
                        "allow_redirects": False,
                    },
                ),
                model=model,
                vertexai_config=VertexAIConfig(
                    credentials=google_creds, project=project, location=location
                ),
                **config,
            )

            llm._project = project
            llm._location = location
            llm._credentials = credentials
            return llm
        case LLMProvider.GITEEAI:
            from llama_index.llms.openai_like import OpenAILike

            config.setdefault("is_chat_model", True)
            config.setdefault("context_window", 128 * 1024)
            return OpenAILike(
                **openai_http_clients(),
                model=model,
                api_key=credentials,
                **config,
            )
        case LLMProvider.AZURE_OPENAI:
            from llama_index.llms.azure_openai import AzureOpenAI

            return AzureOpenAI(
                **openai_http_clients(),
                model=model,
                api_key=credentials,
                **config,
            )
        case LLMProvider.OLLAMA:
            from llama_index.llms.ollama import Ollama

            config.setdefault("request_timeout", 60 * 10)
            config.setdefault("context_window", 8192)
            from ollama import Client, AsyncClient
            from app.security.generation_http import (
                GenerationTransport,
                GenerationAsyncTransport,
            )

            options = {
                "host": config["base_url"],
                "timeout": config["request_timeout"],
                "trust_env": False,
                "follow_redirects": False,
                "headers": {
                    "Authorization": f"Bearer {credentials or 'autoflow-local-no-auth'}"
                },
            }
            return Ollama(
                model=model,
                client=Client(
                    transport=GenerationTransport(trust_env=False), **options
                ),
                async_client=AsyncClient(
                    transport=GenerationAsyncTransport(trust_env=False), **options
                ),
                **config,
            )
        case _:
            raise ValueError(f"Got unknown LLM provider: {provider}")


def get_llm_by_id(session: Session, llm_id: int) -> Optional[LLM]:
    db_llm = llm_repo.get(session, llm_id)
    if not db_llm:
        return None
    return resolve_llm(
        db_llm.provider,
        db_llm.model,
        db_llm.config,
        db_llm.credentials,
    )


def must_get_llm_by_id(session: Session, llm_id: int) -> LLM:
    db_llm = llm_repo.must_get(session, llm_id)
    return resolve_llm(
        db_llm.provider,
        db_llm.model,
        db_llm.config,
        db_llm.credentials,
    )


def get_default_llm(session: Session) -> Optional[LLM]:
    db_llm = llm_repo.get_default(session)
    if not db_llm:
        return None
    return resolve_llm(
        db_llm.provider,
        db_llm.model,
        db_llm.config,
        db_llm.credentials,
    )


def must_get_default_llm(session: Session) -> LLM:
    db_llm = llm_repo.must_get_default(session)
    return resolve_llm(
        db_llm.provider,
        db_llm.model,
        db_llm.config,
        db_llm.credentials,
    )


def get_llm_or_default(session: Session, llm_id: Optional[int]) -> LLM:
    if llm_id is None:
        return must_get_default_llm(session)
    else:
        return must_get_llm_by_id(session, llm_id)
