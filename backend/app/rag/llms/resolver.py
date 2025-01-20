import os
from typing import Optional
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.gemini import Gemini
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.bedrock.utils import BEDROCK_FOUNDATION_LLMS
from llama_index.llms.ollama import Ollama
from llama_index.llms.vertex import Vertex
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from sqlmodel import Session

from app.repositories.llm import llm_repo
from app.rag.llms.provider import LLMProvider


def resolve_llm(
    provider: LLMProvider,
    model: str,
    config: dict,
    credentials: str | list | dict | None,
) -> LLM:
    match provider:
        case LLMProvider.OPENAI:
            return OpenAI(
                model=model,
                api_key=credentials,
                **config,
            )
        case LLMProvider.OPENAI_LIKE:
            config.setdefault("context_window", 200 * 1000)
            return OpenAILike(model=model, api_key=credentials, **config)
        case LLMProvider.GEMINI:
            os.environ["GOOGLE_API_KEY"] = credentials
            return Gemini(model=model, api_key=credentials, **config)
        case LLMProvider.BEDROCK:
            access_key_id = credentials["aws_access_key_id"]
            secret_access_key = credentials["aws_secret_access_key"]
            region_name = credentials["aws_region_name"]

            context_size = None
            if model not in BEDROCK_FOUNDATION_LLMS:
                context_size = 200000

            llm = Bedrock(
                model=model,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name,
                context_size=context_size,
                **config,
            )
            # Note: Because llama index Bedrock class doesn't set up these values to the corresponding
            # attributes in its constructor function, we pass the values again via setter to pass them to
            # `get_dspy_lm_by_llama_llm` function.
            llm.aws_access_key_id = access_key_id
            llm.aws_secret_access_key = secret_access_key
            llm.region_name = region_name
            return llm
        case LLMProvider.ANTHROPIC_VERTEX:
            google_creds: service_account.Credentials = (
                service_account.Credentials.from_service_account_info(
                    credentials,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            )
            google_creds.refresh(request=Request())
            if "max_tokens" not in config:
                config.update(max_tokens=4096)
            return Vertex(
                model=model,
                credentials=google_creds,
                **config,
            )
        case LLMProvider.OLLAMA:
            config.setdefault("request_timeout", 60 * 10)
            config.setdefault("context_window", 4096)
            return Ollama(model=model, **config)
        case LLMProvider.GITEEAI:
            config.setdefault("context_window", 200 * 1000)
            return OpenAILike(
                model=model,
                api_base="https://ai.gitee.com/v1",
                api_key=credentials,
                **config,
            )
        case _:
            raise ValueError(f"Got unknown LLM provider: {provider}")


def get_llm(session: Session, llm_id: int) -> Optional[LLM]:
    db_llm = llm_repo.get(session, llm_id)
    if not db_llm:
        return None
    return resolve_llm(
        db_llm.provider,
        db_llm.model,
        db_llm.config,
        db_llm.credentials,
    )


def must_get_llm(session: Session, llm_id: int) -> LLM:
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
