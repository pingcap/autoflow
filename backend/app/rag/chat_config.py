import logging
import dspy

from typing import Dict, Optional
from pydantic import BaseModel
from sqlmodel import Session

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.llms.llm import LLM

from app.utils.dspy import get_dspy_lm_by_llama_llm
from app.rag.llms.resolver import get_default_llm, resolve_llm
from app.rag.rerankers.resolver import get_default_reranker_model, resolve_reranker
from app.rag.postprocessors import get_metadata_post_filter, MetadataFilters
from app.repositories import knowledge_base_repo

from app.models import (
    ChatEngine as DBChatEngine,
    LLM as DBLLM,
    RerankerModel as DBRerankerModel,
    KnowledgeBase,
)
from app.repositories import chat_engine_repo
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
    relationship_meta_filters: Optional[Dict] = None


class ExternalChatEngine(BaseModel):
    # TODO: add enable flag for this config.
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

    def get_linked_knowledge_base(self, session: Session) -> KnowledgeBase | None:
        if not self.knowledge_base:
            return None
        return knowledge_base_repo.must_get(
            session, self.knowledge_base.linked_knowledge_base.id
        )

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
        return resolve_llm(
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
        return resolve_llm(
            self._db_fast_llm.provider,
            self._db_fast_llm.model,
            self._db_fast_llm.config,
            self._db_fast_llm.credentials,
        )

    def get_fast_dspy_lm(self, session: Session) -> dspy.LM:
        llama_llm = self.get_fast_llama_llm(session)
        return get_dspy_lm_by_llama_llm(llama_llm)

    # FIXME: Reranker top_n should be config in the retrival config.
    def get_reranker(
        self, session: Session, top_n: int = None
    ) -> Optional[BaseNodePostprocessor]:
        if not self._db_reranker:
            return get_default_reranker_model(session, top_n)

        top_n = self._db_reranker.top_n if top_n is None else top_n
        return resolve_reranker(
            self._db_reranker.provider,
            self._db_reranker.model,
            top_n,
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
