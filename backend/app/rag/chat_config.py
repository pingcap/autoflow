import os
import logging
from typing import Optional

import dspy
from llama_index.llms.bedrock.utils import BEDROCK_FOUNDATION_LLMS
from pydantic import BaseModel
from llama_index.llms.openai.utils import DEFAULT_OPENAI_API_BASE
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.gemini import Gemini
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.ollama import Ollama
from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank
from sqlmodel import Session, select
from google.oauth2 import service_account
from google.auth.transport.requests import Request

from app.rag.embeddings.openai_like_embedding import OpenAILikeEmbedding
from app.rag.node_postprocessor import MetadataPostFilter
from app.rag.node_postprocessor.metadata_post_filter import MetadataFilters
from app.rag.node_postprocessor.baisheng_reranker import BaishengRerank
from app.rag.node_postprocessor.local_reranker import LocalRerank
from app.rag.embeddings.local_embedding import LocalEmbedding
from app.repositories import chat_engine_repo
from app.repositories.embedding_model import embedding_model_repo
from app.repositories.llm import get_default_db_llm
from app.types import LLMProvider, EmbeddingProvider, RerankerProvider
from app.rag.default_prompt import (
    DEFAULT_INTENT_GRAPH_KNOWLEDGE,
    DEFAULT_NORMAL_GRAPH_KNOWLEDGE,
    DEFAULT_CONDENSE_QUESTION_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
    DEFAULT_REFINE_PROMPT,
    DEFAULT_FURTHER_QUESTIONS_PROMPT,
    DEFAULT_CONDENSE_ANSWER_PROMPT,
    DEFAULT_GENERATE_GOAL_PROMPT,
    DEFAULT_CLARIFYING_QUESTION_PROMPT,
)
from app.models import (
    ChatEngine as DBChatEngine,
    LLM as DBLLM,
    RerankerModel as DBRerankerModel,
)

from app.rag.llms.anthropic_vertex import AnthropicVertex
from app.utils.dspy import get_dspy_lm_by_llama_llm

logger = logging.getLogger(__name__)


class LLMOption(BaseModel):
    intent_graph_knowledge: str = DEFAULT_INTENT_GRAPH_KNOWLEDGE
    normal_graph_knowledge: str = DEFAULT_NORMAL_GRAPH_KNOWLEDGE
    condense_question_prompt: str = DEFAULT_CONDENSE_QUESTION_PROMPT
    clarifying_question_prompt: str = DEFAULT_CLARIFYING_QUESTION_PROMPT
    text_qa_prompt: str = DEFAULT_TEXT_QA_PROMPT
    refine_prompt: str = DEFAULT_REFINE_PROMPT
    further_questions_prompt: str = DEFAULT_FURTHER_QUESTIONS_PROMPT
    condense_answer_prompt: str = DEFAULT_CONDENSE_ANSWER_PROMPT
    generate_goal_prompt: str = DEFAULT_GENERATE_GOAL_PROMPT

class VectorSearchOption(BaseModel):
    metadata_post_filters: Optional[MetadataFilters] = None


class KnowledgeGraphOption(BaseModel):
    enabled: bool = True
    depth: int = 2
    include_meta: bool = True
    with_degree: bool = False
    using_intent_search: bool = True
    relationship_meta_filters: Optional[dict] = None


class ExternalChatEngine(BaseModel):
    stream_chat_api_url: str = None


class LinkedKnowledgeBase(BaseModel):
    id: int


class KnowledgeBaseOption(BaseModel):
    linked_knowledge_base: LinkedKnowledgeBase
    # TODO: Support multiple knowledge base retrieve.
    # linked_knowledge_bases: List[LinkedKnowledgeBase]


class ChatEngineConfig(BaseModel):
    llm: LLMOption = LLMOption()
    # Notice: Currently knowledge base option is optional, if it is not configured, it will use
    # the deprecated chunks / relationships / entities table as the data source.
    knowledge_base: Optional[KnowledgeBaseOption] = None
    knowledge_graph: KnowledgeGraphOption = KnowledgeGraphOption()
    vector_search: VectorSearchOption = VectorSearchOption()
    post_verification_url: Optional[str] = None
    post_verification_token: Optional[str] = None
    external_engine_config: Optional[ExternalChatEngine] = None
    hide_sources: bool = False
    clarify_question: bool = False

    _db_chat_engine: Optional[DBChatEngine] = None
    _db_llm: Optional[DBLLM] = None
    _db_fast_llm: Optional[DBLLM] = None
    _db_reranker: Optional[DBRerankerModel] = None

    def get_db_chat_engine(self) -> Optional[DBChatEngine]:
        return self._db_chat_engine

    @classmethod
    def load_from_db(cls, session: Session, engine_name: str) -> "ChatEngineConfig":
        if not engine_name or engine_name == "default":
            db_chat_engine = chat_engine_repo.get_default_engine(session)
        else:
            db_chat_engine = chat_engine_repo.get_engine_by_name(session, engine_name)

        if not db_chat_engine:
            logger.warning(
                f"Chat engine {engine_name} not found in DB, using default engine"
            )
            db_chat_engine = chat_engine_repo.get_default_engine(session)

        obj = cls.model_validate(db_chat_engine.engine_options)
        obj._db_chat_engine = db_chat_engine
        obj._db_llm = db_chat_engine.llm
        obj._db_fast_llm = db_chat_engine.fast_llm
        obj._db_reranker = db_chat_engine.reranker
        return obj

    def get_llama_llm(self, session: Session) -> LLM:
        if not self._db_llm:
            return get_default_llm(session)
        return get_llm(
            self._db_llm.provider,
            self._db_llm.model,
            self._db_llm.config,
            self._db_llm.credentials,
        )

    def get_dspy_lm(self, session: Session) -> dspy.LM:
        llama_llm = self.get_llama_llm(session)
        return get_dspy_lm_by_llama_llm(llama_llm)

    def get_fast_llama_llm(self, session: Session) -> LLM:
        if not self._db_fast_llm:
            return get_default_llm(session)
        return get_llm(
            self._db_fast_llm.provider,
            self._db_fast_llm.model,
            self._db_fast_llm.config,
            self._db_fast_llm.credentials,
        )

    def get_fast_dspy_lm(self, session: Session) -> dspy.LM:
        llama_llm = self.get_fast_llama_llm(session)
        return get_dspy_lm_by_llama_llm(llama_llm)

    def get_reranker(self, session: Session) -> Optional[BaseNodePostprocessor]:
        if not self._db_reranker:
            return get_default_reranker_model(session)
        return get_reranker_model(
            self._db_reranker.provider,
            self._db_reranker.model,
            self._db_reranker.top_n,
            self._db_reranker.config,
            self._db_reranker.credentials,
        )

    def get_metadata_filter(self) -> BaseNodePostprocessor:
        return get_metadata_post_filter(self.vector_search.metadata_post_filters)

    def screenshot(self) -> dict:
        return self.model_dump(
            exclude={
                "llm": [
                    "condense_question_prompt",
                    "text_qa_prompt",
                    "refine_prompt",
                ],
                "post_verification_token": True,
            }
        )


def get_llm(
    provider: LLMProvider,
    model: str,
    config: dict,
    credentials: str | list | dict | None,
) -> LLM:
    match provider:
        case LLMProvider.OPENAI:
            api_base = config.pop("api_base", DEFAULT_OPENAI_API_BASE)
            return OpenAI(
                model=model,
                api_base=api_base,
                api_key=credentials,
                **config,
            )
        case LLMProvider.OPENAI_LIKE:
            llm = OpenAILike(model=model, api_key=credentials, **config)
            llm.context_window = 200000
            return llm
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
            return AnthropicVertex(model=model, credentials=google_creds, **config)
        case LLMProvider.OLLAMA:
            config.setdefault("request_timeout", 60 * 10)
            config.setdefault("context_window", 4096)
            return Ollama(model=model, **config)
        case _:
            raise ValueError(f"Got unknown LLM provider: {provider}")


def get_default_llm(session: Session) -> LLM:
    db_llm = get_default_db_llm(session)
    if not db_llm:
        raise ValueError("No default LLM found in DB")
    return get_llm(
        db_llm.provider,
        db_llm.model,
        db_llm.config,
        db_llm.credentials,
    )


def get_embedding_model(
    provider: EmbeddingProvider,
    model: str,
    config: dict,
    credentials: str | list | dict | None,
) -> BaseEmbedding:
    match provider:
        case EmbeddingProvider.OPENAI:
            api_base = config.pop("api_base", DEFAULT_OPENAI_API_BASE)
            return OpenAIEmbedding(
                model=model,
                api_base=api_base,
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
        case EmbeddingProvider.OPENAI_LIKE:
            api_base = config.pop("api_base", "https://open.bigmodel.cn/api/paas/v4")
            return OpenAILikeEmbedding(
                model=model,
                api_base=api_base,
                api_key=credentials,
                **config,
            )
        case _:
            raise ValueError(f"Got unknown embedding provider: {provider}")


def get_default_embedding_model(session: Session) -> BaseEmbedding:
    db_embed_model = embedding_model_repo.get_default_model(session)
    if not db_embed_model:
        raise ValueError("No default embedding model found in DB")
    return get_embedding_model(
        db_embed_model.provider,
        db_embed_model.model,
        db_embed_model.config,
        db_embed_model.credentials,
    )


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
            )
        case RerankerProvider.COHERE:
            return CohereRerank(
                model=model,
                top_n=top_n,
                api_key=credentials,
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
        case _:
            raise ValueError(f"Got unknown reranker provider: {provider}")


def get_default_reranker_model(session: Session) -> Optional[BaseNodePostprocessor]:
    db_reranker = session.exec(
        select(DBRerankerModel).order_by(DBRerankerModel.is_default.desc()).limit(1)
    ).first()
    if not db_reranker:
        return None
    return get_reranker_model(
        db_reranker.provider,
        db_reranker.model,
        db_reranker.top_n,
        db_reranker.config,
        db_reranker.credentials,
    )


def get_metadata_post_filter(
    filters: Optional[MetadataFilters] = None,
) -> BaseNodePostprocessor:
    return MetadataPostFilter(filters)
