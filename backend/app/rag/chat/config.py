import logging
import dspy

from typing import Optional, List
from pydantic import BaseModel, Field
from sqlmodel import Session

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.llms.llm import LLM

from app.rag.postprocessors.metadata_post_filter import MetadataPostFilter
from app.rag.retrievers.chunk.schema import VectorSearchRetrieverConfig
from app.rag.retrievers.knowledge_graph.schema import KnowledgeGraphRetrieverConfig
from app.rag.llms.dspy import get_dspy_lm_by_llama_llm
from app.rag.llms.resolver import get_default_llm, resolve_llm
from app.rag.rerankers.resolver import get_default_reranker_model, resolve_reranker

from app.models import (
    LLM as DBLLM,
    RerankerModel as DBRerankerModel,
    KnowledgeBase,
    ChatEngine as DBChatEngine,
)
from app.repositories import chat_engine_repo, knowledge_base_repo
from app.rag.default_prompt import (
    DEFAULT_INTENT_GRAPH_KNOWLEDGE,
    DEFAULT_NORMAL_GRAPH_KNOWLEDGE,
    DEFAULT_CONDENSE_QUESTION_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
    DEFAULT_FURTHER_QUESTIONS_PROMPT,
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
    further_questions_prompt: str = DEFAULT_FURTHER_QUESTIONS_PROMPT
    generate_goal_prompt: str = DEFAULT_GENERATE_GOAL_PROMPT


class VectorSearchOption(VectorSearchRetrieverConfig):
    pass


class KnowledgeGraphOption(KnowledgeGraphRetrieverConfig):
    enabled: bool = True
    using_intent_search: bool = True


class ExternalChatEngine(BaseModel):
    # TODO: add enable flag for this config.
    stream_chat_api_url: str = None


class LinkedKnowledgeBase(BaseModel):
    id: int


class KnowledgeBaseOption(BaseModel):
    linked_knowledge_base: LinkedKnowledgeBase = None
    linked_knowledge_bases: Optional[List[LinkedKnowledgeBase]] = Field(
        default_factory=list
    )


class ChatEngineConfig(BaseModel):
    external_engine_config: Optional[ExternalChatEngine] = None

    llm: LLMOption = LLMOption()

    knowledge_base: KnowledgeBaseOption = KnowledgeBaseOption()
    knowledge_graph: KnowledgeGraphOption = KnowledgeGraphOption()
    vector_search: VectorSearchOption = VectorSearchOption()

    refine_question_with_kg: bool = True
    clarify_question: bool = False
    further_questions: bool = False

    post_verification_url: Optional[str] = None
    post_verification_token: Optional[str] = None
    hide_sources: bool = False

    _db_chat_engine: Optional[DBChatEngine] = None
    _db_llm: Optional[DBLLM] = None
    _db_fast_llm: Optional[DBLLM] = None
    _db_reranker: Optional[DBRerankerModel] = None

    @property
    def is_external_engine(self) -> bool:
        return (
            self.external_engine_config is not None
            and self.external_engine_config.stream_chat_api_url
        )

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

    # FIXME: Reranker top_n should be config in the retrieval config.
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
        return MetadataPostFilter(self.vector_search.metadata_filters)

    def get_knowledge_bases(self, db_session: Session) -> List[KnowledgeBase]:
        if not self.knowledge_base:
            return []
        kb_config: KnowledgeBaseOption = self.knowledge_base
        linked_knowledge_base_ids = []
        if len(kb_config.linked_knowledge_bases) == 0:
            linked_knowledge_base_ids.append(kb_config.linked_knowledge_base.id)
        else:
            linked_knowledge_base_ids.extend(
                [kb.id for kb in kb_config.linked_knowledge_bases]
            )
        knowledge_bases = knowledge_base_repo.get_by_ids(
            db_session, knowledge_base_ids=linked_knowledge_base_ids
        )
        return knowledge_bases

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
