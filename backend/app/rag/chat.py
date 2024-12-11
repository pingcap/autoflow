import json
import time
import logging

import dspy

from uuid import UUID
from typing import List, Generator, Optional, Tuple, Callable, Type
from datetime import datetime, UTC
from urllib.parse import urljoin

import requests
import jinja2
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from pydantic import BaseModel
from sqlalchemy import text, delete
from sqlmodel import Session, select, func, SQLModel
from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.base import ChatMessage
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.callbacks.schema import EventPayload
from llama_index.core.callbacks import CallbackManager
from llama_index.core.response_synthesizers import get_response_synthesizer
from langfuse import Langfuse
from langfuse.llama_index import LlamaIndexCallbackHandler

from app.api.routes.models import (
    RequiredConfigStatus,
    OptionalConfigStatus,
    NeedMigrationStatus,
)
from app.models import (
    User,
    Document as DBDocument,
    ChatVisibility,
    Chat as DBChat,
    ChatMessage as DBChatMessage,
    KnowledgeBase as DBKnowledgeBase,
    RerankerModel as DBRerankerModel,
    Chunk as DBChunk,
    Entity as DBEntity,
    Relationship as DBRelationship,
    ChatEngine,
)
from app.core.config import settings
from app.models.chunk import get_kb_chunk_model
from app.models.entity import get_kb_entity_model
from app.models.recommend_question import RecommendQuestion
from app.rag.chat_stream_protocol import (
    ChatStreamMessagePayload,
    ChatStreamDataPayload,
    ChatEvent,
)
from app.models.relationship import get_kb_relationship_model
from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.knowledge_base.index_store import get_kb_tidb_graph_editor
from app.rag.knowledge_graph.graph_store import TiDBGraphStore
from app.rag.vector_store.tidb_vector_store import TiDBVectorStore
from app.rag.knowledge_graph.graph_store.tidb_graph_editor import (
    TiDBGraphEditor,
    legacy_tidb_graph_editor,
)

from app.rag.knowledge_graph import KnowledgeGraphIndex
from app.rag.chat_config import (
    ChatEngineConfig,
    get_default_embed_model,
    KnowledgeGraphOption,
    must_get_default_embed_model,
)
from app.rag.types import (
    MyCBEventType,
    ChatMessageSate,
    ChatEventType,
    MessageRole,
)
from app.repositories import chat_repo, knowledge_base_repo, chat_engine_repo
from app.repositories.embedding_model import embed_model_repo
from app.repositories.llm import llm_repo
from app.site_settings import SiteSetting
from app.exceptions import ChatNotFound

logger = logging.getLogger(__name__)


class ChatService:
    _chunk_db_model: Type[SQLModel] = DBChunk
    _entity_db_model: Type[SQLModel] = DBEntity
    _relationship_db_model: Type[SQLModel] = DBRelationship

    def __init__(
        self,
        *,
        db_session: Session,
        user: User,
        browser_id: str,
        origin: str,
        chat_messages: List[ChatMessage],
        engine_name: str = "default",
        chat_id: Optional[UUID] = None,
    ) -> None:
        self.db_session = db_session
        self.user = user
        self.browser_id = browser_id
        self.engine_name = engine_name

        self.user_question, self.chat_history = self._parse_chat_messages(chat_messages)
        if chat_id:
            # FIXME:
            #   only chat owner or superuser can access the chat,
            #   anonymous user can only access anonymous chat by track_id
            self.db_chat_obj = chat_repo.get(self.db_session, chat_id)
            if not self.db_chat_obj:
                raise ChatNotFound()
            try:
                self.chat_engine_config = ChatEngineConfig.load_from_db(
                    db_session, self.db_chat_obj.engine.name
                )
                self.db_chat_engine = self.chat_engine_config.get_db_chat_engine()
            except Exception as e:
                logger.error(f"Failed to load chat engine config: {e}")
                self.chat_engine_config = ChatEngineConfig.load_from_db(
                    db_session, engine_name
                )
                self.db_chat_engine = self.chat_engine_config.get_db_chat_engine()
            logger.info(
                f"ChatService - chat_id: {chat_id}, chat_engine: {self.db_chat_obj.engine.name}"
            )
            self.chat_history = [
                ChatMessage(role=m.role, content=m.content, additional_kwargs={})
                for m in chat_repo.get_messages(self.db_session, self.db_chat_obj)
            ]
        else:
            self.chat_engine_config = ChatEngineConfig.load_from_db(
                db_session, engine_name
            )
            self.db_chat_engine = self.chat_engine_config.get_db_chat_engine()
            self.db_chat_obj = chat_repo.create(
                self.db_session,
                DBChat(
                    title=self.user_question[:100],
                    engine_id=self.db_chat_engine.id,
                    engine_options=self.chat_engine_config.screenshot(),
                    user_id=self.user.id if self.user else None,
                    browser_id=self.browser_id,
                    origin=origin,
                    visibility=ChatVisibility.PUBLIC.value
                    if not self.user
                    else ChatVisibility.PRIVATE.value,
                ),
            )
            chat_id = self.db_chat_obj.id
            # slack/discord may create a new chat with history messages
            now = datetime.now(UTC)
            for i, m in enumerate(self.chat_history):
                chat_repo.create_message(
                    session=self.db_session,
                    chat=self.db_chat_obj,
                    chat_message=DBChatMessage(
                        role=m.role,
                        content=m.content,
                        ordinal=i + 1,
                        created_at=now,
                        updated_at=now,
                        finished_at=now,
                    ),
                )

        self._reranker = self.chat_engine_config.get_reranker(db_session)
        self._metadata_filter = self.chat_engine_config.get_metadata_filter()
        if self._reranker:
            self._node_postprocessors = [self._metadata_filter, self._reranker]
            # Set initial similarity_top_k to a large number,
            # reranker will filter out irrelevant nodes after the retrieval
            self._similarity_top_k = 60
        else:
            self._node_postprocessors = [self._metadata_filter]
            self._similarity_top_k = 10

        # Langfuse
        self.langfuse_host = SiteSetting.langfuse_host
        self.langfuse_secret_key = SiteSetting.langfuse_secret_key
        self.langfuse_public_key = SiteSetting.langfuse_public_key
        self.enable_langfuse = (
            self.langfuse_host and self.langfuse_secret_key and self.langfuse_public_key
        )

        # TODO: Support multiple knowledge base retrieve.
        if self.chat_engine_config.knowledge_base:
            linked_knowledge_base = (
                self.chat_engine_config.knowledge_base.linked_knowledge_base
            )
            kb = knowledge_base_repo.must_get(db_session, linked_knowledge_base.id)
            self._embed_model = get_kb_embed_model(db_session, kb)
            self._chunk_db_model = get_kb_chunk_model(kb)
            self._entity_db_model = get_kb_entity_model(kb)
            self._relationship_db_model = get_kb_relationship_model(kb)
        else:
            self._embed_model = get_default_embed_model(db_session)

    def chat(self) -> Generator[ChatEvent | str, None, None]:
        try:
            if (
                self.chat_engine_config.external_engine_config
                and self.chat_engine_config.external_engine_config.stream_chat_api_url
            ):
                for event in self._external_chat():
                    yield event
            else:
                for event in self._chat():
                    yield event
        except Exception as e:
            logger.exception(e)
            yield ChatEvent(
                event_type=ChatEventType.ERROR_PART,
                payload="Encountered an error while processing the chat. Please try again later.",
            )

    def _search_kg(
        self,
        kg_config: KnowledgeGraphOption,
        fast_dspy_lm: dspy.LM,
        embed_model: BaseEmbedding,
        get_llamaindex_callback_manager: Callable[[], Optional[CallbackManager]],
        trace_url: str,
        annotation_silent: bool = False,
    ) -> Generator[
        ChatEvent | str, None, Tuple[List[dict], List[dict], List[dict], dict, str]
    ]:
        """
        Search the knowledge graph for relevant entities, relationships, and chunks.
        Args:
            kg_config: KnowledgeGraphOption
            fast_dspy_lm: dspy.LM
            embed_model: BaseEmbedding
            get_llamaindex_callback_manager: Callable[[], CallbackManager]
            trace_url: str
            annotation_silent: bool, if True, do not send annotation events

        Returns:
            List[dict]: entities
            List[dict]: relationships
            List[dict]: chunks
            dict: graph_data_source_ids
            str: graph_knowledges_context
        """

        entities, relations, chunks = [], [], []
        graph_data_source_ids = {}
        graph_knowledges_context = ""

        if kg_config.enabled:
            graph_store = TiDBGraphStore(
                dspy_lm=fast_dspy_lm,
                session=self.db_session,
                embed_model=embed_model,
                entity_db_model=self._entity_db_model,
                relationship_db_model=self._relationship_db_model,
            )
            graph_index: KnowledgeGraphIndex = KnowledgeGraphIndex.from_existing(
                dspy_lm=fast_dspy_lm,
                kg_store=graph_store,
                callback_manager=get_llamaindex_callback_manager(),
            )

            if kg_config.using_intent_search:
                if not annotation_silent:
                    yield ChatEvent(
                        event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                        payload=ChatStreamMessagePayload(
                            state=ChatMessageSate.KG_RETRIEVAL,
                            display="Identifying Your Question's Core Intents",
                        ),
                    )
                graph_index._callback_manager = get_llamaindex_callback_manager()
                sub_queries = graph_index.intent_analyze(
                    self.user_question,
                    self.chat_history,
                )
                if not annotation_silent:
                    yield ChatEvent(
                        event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                        payload=ChatStreamMessagePayload(
                            state=ChatMessageSate.TRACE,
                            display="Searching the Knowledge Graph for Relevant Context",
                            context={"langfuse_url": trace_url},
                        ),
                    )
                graph_index._callback_manager = get_llamaindex_callback_manager()
                result = graph_index.graph_semantic_search(
                    sub_queries,
                    include_meta=True,
                    relationship_meta_filters=kg_config.relationship_meta_filters,
                )

                entities = result["graph"]["entities"]
                relations = result["graph"]["relationships"]
                graph_data_source_ids = {
                    "entities": [e["id"] for e in entities],
                    "relationships": [r["id"] for r in relations],
                }

                graph_knowledges = get_prompt_by_jinja2_template(
                    self.chat_engine_config.llm.intent_graph_knowledge,
                    sub_queries=result["queries"],
                )
                graph_knowledges_context = graph_knowledges.template
            else:
                if not annotation_silent:
                    yield ChatEvent(
                        event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                        payload=ChatStreamMessagePayload(
                            state=ChatMessageSate.TRACE,
                            display="Searching the Knowledge Graph for Relevant Context",
                            context={"langfuse_url": trace_url},
                        ),
                    )
                graph_index._callback_manager = get_llamaindex_callback_manager()
                entities, relations, chunks = graph_index.retrieve_with_weight(
                    self.user_question,
                    [],
                    depth=kg_config.depth,
                    include_meta=kg_config.include_meta,
                    with_degree=kg_config.with_degree,
                    relationship_meta_filters=kg_config.relationship_meta_filters,
                    with_chunks=False,
                )
                graph_data_source_ids = {
                    "entities": [e["id"] for e in entities],
                    "relationships": [r["id"] for r in relations],
                }
                graph_knowledges = get_prompt_by_jinja2_template(
                    self.chat_engine_config.llm.normal_graph_knowledge,
                    entities=entities,
                    relationships=relations,
                )
                graph_knowledges_context = graph_knowledges.template

        return (
            entities,
            relations,
            chunks,
            graph_data_source_ids,
            graph_knowledges_context,
        )

    def _get_llamaindex_callback_manager(
        self,
        langfuse: Optional[Langfuse] = None,
        trace_id: Optional[str] = None,
        llm: Optional[LLM] = None,
        fast_llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
    ) -> CallbackManager:
        # Why we don't use high-level decorator `observe()` as \
        #   `https://langfuse.com/docs/integrations/llama-index/get-started` suggested?
        # track:
        #   - https://github.com/langfuse/langfuse/issues/2015
        #   - https://langfuse.com/blog/2024-04-python-decorator
        if self.enable_langfuse and langfuse and trace_id:
            observation = langfuse.trace(id=trace_id)
            langfuse_handler = LlamaIndexCallbackHandler()
            langfuse_handler.set_root(observation)
            callback_manager = CallbackManager([langfuse_handler])
        else:
            callback_manager = CallbackManager([])

        if llm:
            llm.callback_manager = callback_manager
        if fast_llm:
            fast_llm.callback_manager = callback_manager
        if embed_model:
            embed_model.callback_manager = callback_manager

        return callback_manager

    def _get_langfuse_config(self):
        langfuse = None
        trace_id = ""
        trace_url = ""

        if self.enable_langfuse:
            langfuse = Langfuse(
                host=self.langfuse_host,
                secret_key=self.langfuse_secret_key,
                public_key=self.langfuse_public_key,
            )
            observation = langfuse.trace(
                name="chat",
                user_id=self.user.email
                if self.user
                else f"anonymous-{self.browser_id}",
                metadata={
                    "chat_engine_config": self.chat_engine_config.screenshot(),
                },
                tags=[f"chat_engine:{self.engine_name}"],
                release=settings.ENVIRONMENT,
                input={
                    "user_question": self.user_question,
                    "chat_history": self.chat_history,
                },
            )
            trace_id = observation.trace_id
            trace_url = observation.get_trace_url()

        return langfuse, trace_id, trace_url

    class ClarityResult(BaseModel):
        clarity_needed: bool
        clarifying_question: str

    def _refine_or_early_stop(
        self,
        get_llamaindex_callback_manager: Callable[[], Optional[CallbackManager]],
        fast_llm: LLM,
        graph_knowledges_context: str,
        refined_question_prompt: Optional[str] = None,
        annotation_silent: bool = False,
    ) -> Generator[ChatEvent | str, None, Tuple[bool, str, str]]:
        """
        Determine whether to refine the user question or early stop the conversation with a clarifying question.

        Args:
            get_llamaindex_callback_manager: Callable[[], CallbackManager]
            fast_llm: LLM
            graph_knowledges_context: str
            refined_question_prompt: Optional[str], if it's None, use the default condense_question_prompt
            annotation_silent: bool, if True, do not send annotation events

        Returns:
            bool: whether to early stop the conversation
            str: clarifying question
            str: refined question
        """
        if refined_question_prompt is None:
            refined_question_prompt = (
                self.chat_engine_config.llm.condense_question_prompt
            )

        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.REFINE_QUESTION,
                    display="Query Rewriting for Enhanced Information Retrieval",
                ),
            )
        callback_manager = get_llamaindex_callback_manager()

        # 1. Refine the question
        with callback_manager.as_trace("condense_question"):
            with callback_manager.event(
                MyCBEventType.CONDENSE_QUESTION,
                payload={EventPayload.QUERY_STR: self.user_question},
            ) as event:
                refined_question = fast_llm.predict(
                    get_prompt_by_jinja2_template(
                        refined_question_prompt,
                        graph_knowledges=graph_knowledges_context,
                        chat_history=self.chat_history,
                        question=self.user_question,
                    ),
                )
                event.on_end(payload={EventPayload.COMPLETION: refined_question})
        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.REFINE_QUESTION,
                    message=refined_question,
                ),
            )

        # 2. Check if we have enough information to answer the user question or not
        if self.chat_engine_config.clarify_question:
            with callback_manager.as_trace("check_question"):
                with callback_manager.event(
                    MyCBEventType.CLARIFYING_QUESTION,
                    payload={EventPayload.QUERY_STR: refined_question},
                ) as event:
                    clarity_result = fast_llm.predict(
                        prompt=get_prompt_by_jinja2_template(
                            self.chat_engine_config.llm.clarifying_question_prompt,
                            graph_knowledges=graph_knowledges_context,
                            chat_history=self.chat_history,
                            question=refined_question,
                        ),
                    ).strip().strip(".\"\'!")

                    clarity_needed = clarity_result.lower() != "false"

                    event.on_end(
                        payload={
                            EventPayload.COMPLETION: f"Need Clarification: {clarity_needed}, "
                            f"Clarifying Question: {clarity_result}"
                        }
                    )


                    if clarity_needed:
                        if not annotation_silent:
                            yield ChatEvent(
                                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                                payload=ChatStreamMessagePayload(
                                    state=ChatMessageSate.GENERATE_ANSWER,
                                    display="Need to Ask a Clarifying Question",
                                ),
                            )

                        yield ChatEvent(
                            event_type=ChatEventType.TEXT_PART,
                            payload=clarity_result,
                        )

                        return True, clarity_result, ""

        return False, "", refined_question

    def _gen_answer_via_llama_index(
        self,
        get_llamaindex_callback_manager: Callable[[], Optional[CallbackManager]],
        refined_question: str,
        graph_knowledges_context: str,
        llm: LLM,
        embed_model: BaseEmbedding,
        annotation_silent: bool = False,
    ) -> Generator[ChatEvent | str, None, Tuple[StreamingResponse, List[dict]]]:
        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.SEARCH_RELATED_DOCUMENTS,
                    display="Retrieving and Reranking the Best-Matching Data"
                    if self._reranker
                    else "Retrieving the Most Relevant Data",
                ),
            )
        callback_manager = get_llamaindex_callback_manager()
        text_qa_template = get_prompt_by_jinja2_template(
            self.chat_engine_config.llm.text_qa_prompt,
            current_date=datetime.now().strftime("%Y-%m-%d"),
            graph_knowledges=graph_knowledges_context,
            original_question=self.user_question,
        )
        refine_template = get_prompt_by_jinja2_template(
            self.chat_engine_config.llm.refine_prompt,
            graph_knowledges=graph_knowledges_context,
            original_question=self.user_question,
        )
        vector_store = TiDBVectorStore(
            session=self.db_session, chunk_db_model=self._chunk_db_model
        )
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
            callback_manager=callback_manager,
        )
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=True,
            callback_manager=callback_manager,
        )
        query_engine = vector_index.as_query_engine(
            llm=llm,
            response_synthesizer=response_synthesizer,
            node_postprocessors=self._node_postprocessors,
            similarity_top_k=self._similarity_top_k,
        )
        query_engine.callback_manager = callback_manager
        for _np in self._node_postprocessors:
            _np.callback_manager = callback_manager
        response: StreamingResponse = query_engine.query(refined_question)
        source_documents = self._get_source_documents(response)

        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.SOURCE_NODES,
                    context=source_documents,
                ),
            )
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.GENERATE_ANSWER,
                    display="Generating a Precise Answer with AI",
                ),
            )

        return response, source_documents

    def _chat_finish(
        self,
        db_assistant_message: ChatMessage,
        db_user_message: ChatMessage,
        response_text: str,
        source_documents: List[dict],
        graph_data_source_ids: dict,
        annotation_silent: bool = False,
    ):
        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.FINISHED,
                ),
            )

        post_verification_result_url = self._post_verification(
            self.user_question,
            response_text,
            self.db_chat_obj.id,
            db_assistant_message.id,
        )

        db_assistant_message.sources = source_documents
        db_assistant_message.graph_data = graph_data_source_ids
        db_assistant_message.content = response_text
        db_assistant_message.post_verification_result_url = post_verification_result_url
        db_assistant_message.updated_at = datetime.now(UTC)
        db_assistant_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_assistant_message)
        db_user_message.graph_data = graph_data_source_ids
        db_user_message.updated_at = datetime.now(UTC)
        db_user_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_user_message)
        self.db_session.commit()

        yield ChatEvent(
            event_type=ChatEventType.DATA_PART,
            payload=ChatStreamDataPayload(
                chat=self.db_chat_obj,
                user_message=db_user_message,
                assistant_message=db_assistant_message,
            ),
        )

    def _chat(self) -> Generator[ChatEvent | str, None, None]:
        langfuse, trace_id, trace_url = self._get_langfuse_config()

        db_user_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.USER.value,
                trace_url=trace_url,
                content=self.user_question,
            ),
        )
        db_assistant_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.ASSISTANT.value,
                trace_url=trace_url,
                content="",
            ),
        )

        # TODO: remove to __init__?
        _llm = self.chat_engine_config.get_llama_llm(self.db_session)
        _fast_llm = self.chat_engine_config.get_fast_llama_llm(self.db_session)
        _fast_dspy_lm = self.chat_engine_config.get_fast_dspy_lm(self.db_session)

        yield ChatEvent(
            event_type=ChatEventType.DATA_PART,
            payload=ChatStreamDataPayload(
                chat=self.db_chat_obj,
                user_message=db_user_message,
                assistant_message=db_assistant_message,
            ),
        )

        def _get_llamaindex_callback_manager_in_chat() -> CallbackManager:
            return self._get_llamaindex_callback_manager(
                langfuse=langfuse,
                trace_id=trace_id,
                llm=_llm,
                fast_llm=_fast_llm,
                embed_model=self._embed_model,
            )

        # 1. Retrieve entities, relations, and chunks from the knowledge graph
        kg_config = self.chat_engine_config.knowledge_graph
        (
            entities,
            relations,
            chunks,
            graph_data_source_ids,
            graph_knowledges_context,
        ) = yield from self._search_kg(
            kg_config=kg_config,
            fast_dspy_lm=_fast_dspy_lm,
            embed_model=self._embed_model,
            trace_url=trace_url,
            get_llamaindex_callback_manager=_get_llamaindex_callback_manager_in_chat,
        )

        # 2. Refine the user question using graph information and chat history
        # 2.1 Early stop if the user question does not have enough information, we need to ask a clarifying question
        (
            early_stop,
            clarifying_question,
            refined_question,
        ) = yield from self._refine_or_early_stop(
            get_llamaindex_callback_manager=_get_llamaindex_callback_manager_in_chat,
            fast_llm=_fast_llm,
            graph_knowledges_context=graph_knowledges_context,
        )
        if early_stop:
            # the clarifying question is the final response
            response_text = clarifying_question
            source_documents = []
        else:
            # 3. Retrieve the related chunks from the vector store
            # 4. Rerank after the retrieval
            # 5. Generate a response using the refined question and related chunks
            response, source_documents = yield from self._gen_answer_via_llama_index(
                get_llamaindex_callback_manager=_get_llamaindex_callback_manager_in_chat,
                refined_question=refined_question,
                graph_knowledges_context=graph_knowledges_context,
                llm=_llm,
                embed_model=self._embed_model,
            )

            response_text = ""
            for word in response.response_gen:
                response_text += word
                yield ChatEvent(
                    event_type=ChatEventType.TEXT_PART,
                    payload=word,
                )

            if not response_text:
                raise Exception("Got empty response from LLM")

        yield from self._chat_finish(
            db_assistant_message=db_assistant_message,
            db_user_message=db_user_message,
            response_text=response_text,
            source_documents=source_documents,
            graph_data_source_ids=graph_data_source_ids,
        )

    def _external_chat(self) -> Generator[ChatEvent | str, None, None]:
        # TODO: integration with langfuse.
        db_user_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.USER.value,
                trace_url="",
                content=self.user_question,
            ),
        )
        db_assistant_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.ASSISTANT.value,
                trace_url="",
                content="",
            ),
        )

        yield ChatEvent(
            event_type=ChatEventType.DATA_PART,
            payload=ChatStreamDataPayload(
                chat=self.db_chat_obj,
                user_message=db_user_message,
                assistant_message=db_assistant_message,
            ),
        )

        # TODO: remove to __init__?
        _fast_dspy_lm = self.chat_engine_config.get_fast_dspy_lm(self.db_session)
        _fast_llm = self.chat_engine_config.get_fast_llama_llm(self.db_session)

        # retrieve entities, relations, and chunks from the knowledge graph
        # this retrieve progress is only for the clarifying question checking
        try:
            graph_data_source_ids = []
            graph_knowledges_context = ""

            logger.info("start to _refine_or_early_stop")
            (
                early_stop,
                clarifying_question,
                goal,
            ) = yield from self._refine_or_early_stop(
                get_llamaindex_callback_manager=lambda: self._get_llamaindex_callback_manager(
                    fast_llm=_fast_llm,
                    embed_model=self._embed_model,
                ),
                fast_llm=_fast_llm,
                graph_knowledges_context=graph_knowledges_context,
                refined_question_prompt=self.chat_engine_config.llm.generate_goal_prompt,
                annotation_silent=True,
            )
            logger.info(f"start request stackvm with goal: {goal}")
            goal = goal.strip()
            if goal.startswith("Goal: "):
                goal = goal[len("Goal: ") :].strip()

            if early_stop:
                # the clarifying question is the final response
                yield from self._chat_finish(
                    db_assistant_message=db_assistant_message,
                    db_user_message=db_user_message,
                    response_text=clarifying_question,
                    source_documents=[],
                    graph_data_source_ids=graph_data_source_ids,
                    annotation_silent=True,
                )
                return
        except Exception as e:
            logger.error(f"Failed to search kg or refine question: {e}")
            goal = self.user_question

        cache_messages = None
        if settings.ENABLE_QUESTION_CACHE:
            try:
                logger.info(
                    f"start to find_recent_assistant_messages_by_goal with goal: {goal}"
                )
                cache_messages = chat_repo.find_recent_assistant_messages_by_goal(
                    self.db_session, goal
                )
                logger.debug(
                    f"find_recent_assistant_messages_by_goal result: {cache_messages}"
                )
            except Exception as e:
                logger.error(f"Failed to find recent assistant messages by goal: {e}")

        stream_chat_api_url = (
            self.chat_engine_config.external_engine_config.stream_chat_api_url
        )
        if cache_messages and len(cache_messages) > 0:
            stackvm_response_text = cache_messages[0].content
            task_id = cache_messages[0].meta.get("task_id")
            for chunk in stackvm_response_text.split(". "):
                if chunk:
                    if not chunk.endswith("."):
                        chunk += ". "
                    yield ChatEvent(
                        event_type=ChatEventType.TEXT_PART,
                        payload=chunk,
                    )
        else:
            logger.debug(
                f"Chatting with external chat engine (api_url: {stream_chat_api_url}) to answer for user question: {self.user_question}"
            )
            chat_params = {
                "goal": goal,
            }
            res = requests.post(stream_chat_api_url, json=chat_params, stream=True)

            # Notice: External type chat engine doesn't support non-streaming mode for now.
            stackvm_response_text = ""
            task_id = None
            for line in res.iter_lines():
                if not line:
                    continue

                # Append to final response text.
                chunk = line.decode("utf-8")
                if chunk.startswith("0:"):
                    word = json.loads(chunk[2:])
                    stackvm_response_text += word
                    yield ChatEvent(
                        event_type=ChatEventType.TEXT_PART,
                        payload=word,
                    )
                else:
                    yield line + b"\n"

                try:
                    if chunk.startswith("8:") and task_id is None:
                        states = json.loads(chunk[2:])
                        if len(states) > 0:
                            # accesss task by http://endpoint/?task_id=$task_id
                            task_id = states[0].get("task_id")
                except Exception as e:
                    logger.error(f"Failed to get task_id from chunk: {e}")

        response_text = stackvm_response_text
        base_url = stream_chat_api_url.replace("/api/stream_execute_vm", "")
        db_assistant_message.content = response_text
        db_assistant_message.trace_url = (
            f"{base_url}?task_id={task_id}" if task_id else ""
        )
        db_assistant_message.meta = {
            "task_id": task_id,
            "goal": goal,
        }
        db_assistant_message.updated_at = datetime.now(UTC)
        db_assistant_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_assistant_message)
        db_user_message.trace_url = f"{base_url}?task_id={task_id}" if task_id else ""
        db_user_message.meta = {
            "task_id": task_id,
            "goal": goal,
        }
        db_user_message.updated_at = datetime.now(UTC)
        db_user_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_user_message)
        self.db_session.commit()

        yield ChatEvent(
            event_type=ChatEventType.DATA_PART,
            payload=ChatStreamDataPayload(
                chat=self.db_chat_obj,
                user_message=db_user_message,
                assistant_message=db_assistant_message,
            ),
        )

    def _parse_chat_messages(
        self, chat_messages: List[ChatMessage]
    ) -> tuple[str, List[ChatMessage]]:
        user_question = chat_messages[-1].content
        chat_history = chat_messages[:-1]
        return user_question, chat_history

    def _get_source_documents(self, response: StreamingResponse) -> List[dict]:
        source_nodes_ids = [s_n.node_id for s_n in response.source_nodes]
        stmt = (
            select(
                self._chunk_db_model.id,
                DBDocument.id,
                DBDocument.name,
                DBDocument.source_uri,
            )
            .outerjoin(DBDocument, self._chunk_db_model.document_id == DBDocument.id)
            .where(
                self._chunk_db_model.id.in_(source_nodes_ids),
            )
        )
        source_chunks = self.db_session.exec(stmt).all()
        # Sort the source chunks based on the order of the source_nodes_ids, which are arranged according to their related scores.
        source_chunks = sorted(
            source_chunks, key=lambda x: source_nodes_ids.index(str(x[0]))
        )
        source_documents = []
        source_documents_ids = []
        for s in source_chunks:
            if s[1] not in source_documents_ids:
                source_documents_ids.append(s[1])
                source_documents.append(
                    {
                        "id": s[1],
                        "name": s[2],
                        "source_uri": s[3],
                    }
                )
        return source_documents

    def _post_verification(
        self, user_question: str, response_text: str, chat_id: UUID, message_id: int
    ) -> Optional[str]:
        # post verification to external service, will return the post verification result url
        post_verification_url = self.chat_engine_config.post_verification_url
        post_verification_token = self.chat_engine_config.post_verification_token

        if not post_verification_url:
            return

        external_request_id = f"{chat_id}_{message_id}"
        qa_content = f"User question: {user_question}\n\nAnswer:\n{response_text}"
        try:
            resp = requests.post(
                post_verification_url,
                json={
                    "external_request_id": external_request_id,
                    "qa_content": qa_content,
                },
                headers={
                    "Authorization": f"Bearer {post_verification_token}",
                }
                if post_verification_token
                else {},
                timeout=10,
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            return urljoin(f"{post_verification_url}/", str(job_id))
        except Exception:
            logger.exception("Failed to post verification")


def get_prompt_by_jinja2_template(template_string: str, **kwargs) -> PromptTemplate:
    # use jinja2's template because it support complex render logic
    # for example:
    #       {% for e in entities %}
    #           {{ e.name }}
    #       {% endfor %}
    template = (
        jinja2.Template(template_string)
        .render(**kwargs)
        # llama-index will use f-string to format the template
        # so we need to escape the curly braces even if we do not use it
        .replace("{", "{{")
        .replace("}", "}}")
        # This is a workaround to bypass above escape,
        # llama-index will use f-string to format following variables,
        # maybe we can use regex to replace the variable name to make this more robust
        .replace("<<query_str>>", "{query_str}")
        .replace("<<context_str>>", "{context_str}")
        .replace("<<existing_answer>>", "{existing_answer}")
        .replace("<<context_msg>>", "{context_msg}")
    )
    return PromptTemplate(template=template)


def user_can_view_chat(chat: DBChat, user: Optional[User]) -> bool:
    # Anonymous or pulic chat can be accessed by anyone
    # Non-anonymous chat can be accessed by owner or superuser
    if not chat.user_id or chat.visibility == ChatVisibility.PUBLIC:
        return True
    return user is not None and (user.is_superuser or chat.user_id == user.id)


def user_can_edit_chat(chat: DBChat, user: Optional[User]) -> bool:
    if user is None:
        return False
    if user.is_superuser:
        return True
    return chat.user_id == user.id


def get_graph_data_from_chat_message(
    graph_editor: TiDBGraphEditor, session: Session, chat_message: ChatMessage
) -> Tuple[list[dict], list[dict]]:
    if not chat_message.graph_data:
        return [], []

    if "relationships" not in chat_message.graph_data:
        return [], []

    if len(chat_message.graph_data["relationships"]) == 0:
        return [], []

    # FIXME: Why not store the complete data in chat_message.graph_data.
    relationship_ids = chat_message.graph_data["relationships"]
    all_entities, all_relationships = graph_editor.get_relationship_by_ids(
        session, relationship_ids
    )
    entities = [
        {
            "id": e.id,
            "name": e.name,
            "description": e.description,
            "meta": e.meta,
            "entity_type": e.entity_type,
        }
        for e in all_entities
    ]
    relationships = [
        {
            "id": r.id,
            "source_entity_id": r.source_entity_id,
            "target_entity_id": r.target_entity_id,
            "description": r.description,
            "rag_description": f"{r.source_entity.name} -> {r.description} -> {r.target_entity.name}",
            "meta": r.meta,
            "weight": r.weight,
            "last_modified_at": r.last_modified_at,
        }
        for r in all_relationships
    ]

    return entities, relationships


def get_graph_data_from_langfuse(trace_url: str) -> Tuple[list[dict], list[dict]]:
    start_time = time.time()
    langfuse_host = SiteSetting.langfuse_host
    langfuse_secret_key = SiteSetting.langfuse_secret_key
    langfuse_public_key = SiteSetting.langfuse_public_key
    enable_langfuse = langfuse_host and langfuse_secret_key and langfuse_public_key
    current_time = time.time()
    logger.debug(
        f"Graph Load - Fetch langfuse configs from site setting, time cost: {current_time - start_time}s"
    )
    logger.debug(
        f"Graph Load - trace_url: {trace_url}, enable_langfuse: {enable_langfuse}"
    )
    start_time = current_time
    if enable_langfuse and trace_url is not None and trace_url != "":
        langfuse_client = Langfuse(
            secret_key=langfuse_secret_key,
            public_key=langfuse_public_key,
            host=langfuse_host,
        )
        trace_id = trace_url.split("/trace/")[-1]
        ob_data = langfuse_client.fetch_observations(trace_id=trace_id)
        current_time = time.time()
        logger.debug(
            f"Graph Load - Fetch trace({trace_id}) from langfuse, time cost: {current_time - start_time}s"
        )
        start_time = current_time
        all_entities = []
        all_relationships = []

        for obd in ob_data.data:
            if obd.name == MyCBEventType.GRAPH_SEMANTIC_SEARCH:
                for _, sg in obd.output["queries"].items():
                    all_entities.extend(sg["entities"])
                    all_relationships.extend(sg["relationships"])

        unique_entities = {e["id"]: e for e in all_entities}.values()
        unique_relationships = {r["id"]: r for r in all_relationships}.values()

        logger.debug(
            f"Graph Load - Fetch trace({trace_id}) from langfuse, relationships: {len(unique_relationships)}, time cost: {time.time() - start_time}s"
        )

        return list(unique_entities), list(unique_relationships)
    else:
        return [], []


def get_chat_message_subgraph(
    session: Session, chat_message: DBChatMessage
) -> Tuple[List, List]:
    if chat_message.role != MessageRole.USER:
        return [], []

    engine_options = chat_message.chat.engine_options
    chat_engine_config = ChatEngineConfig.model_validate(engine_options)
    kb = chat_engine_config.get_linked_knowledge_base(session)

    # try to get subgraph from chat_message.graph_data
    try:
        graph_editor = (
            get_kb_tidb_graph_editor(session, kb) if kb else legacy_tidb_graph_editor
        )
        entities, relationships = get_graph_data_from_chat_message(
            graph_editor, session, chat_message
        )
        if len(relationships) > 0:
            return list(entities), list(relationships)
    except Exception as e:
        logger.error(f"Failed to get subgraph from chat_message.graph_data: {e}")

    # try to get subgraph from langfuse trace.
    try:
        entities, relationships = get_graph_data_from_langfuse(chat_message.trace_url)
        if len(relationships) > 0:
            return list(entities), list(relationships)
    except Exception as e:
        logger.error(f"Failed to get subgraph from langfuse trace: {e}")

    # try to get subgraph from graph store instead of cached result.

    # Notice: using new chat engine config.
    chat_engine: ChatEngine = chat_message.chat.engine
    chat_engine_config = ChatEngineConfig.load_from_db(session, chat_engine.name)
    kb = chat_engine_config.get_linked_knowledge_base(session)

    embed_model = (
        get_kb_embed_model(session, kb) if kb else must_get_default_embed_model(session)
    )
    entity_db_model = get_kb_entity_model(kb) if kb else DBEntity
    relationship_db_model = get_kb_relationship_model(kb) if kb else DBRelationship
    graph_store = TiDBGraphStore(
        dspy_lm=chat_engine_config.get_fast_dspy_lm(session),
        session=session,
        embed_model=embed_model,
        entity_db_model=entity_db_model,
        relationship_db_model=relationship_db_model,
    )
    kg_config = chat_engine_config.knowledge_graph
    entities, relations, _ = graph_store.retrieve_with_weight(
        chat_message.content,
        [],
        depth=kg_config.depth,
        include_meta=kg_config.include_meta,
        with_degree=kg_config.with_degree,
        with_chunks=False,
    )

    return entities, relations


def check_rag_required_config(session: Session) -> RequiredConfigStatus:
    """
    Check if the required configuration items have been configured, it any of them is
    missing, the RAG application can not complete its work.
    """
    has_default_llm = llm_repo.has_default(session)
    has_default_embedding_model = embed_model_repo.has_default(session)
    has_default_chat_engine = chat_engine_repo.has_default(session)
    has_knowledge_base = session.scalar(select(func.count(DBKnowledgeBase.id))) > 0

    return RequiredConfigStatus(
        default_llm=has_default_llm,
        default_embedding_model=has_default_embedding_model,
        default_chat_engine=has_default_chat_engine,
        knowledge_base=has_knowledge_base,
    )


def check_rag_optional_config(session: Session) -> OptionalConfigStatus:
    langfuse = bool(
        SiteSetting.langfuse_host
        and SiteSetting.langfuse_secret_key
        and SiteSetting.langfuse_public_key
    )
    default_reranker = session.scalar(select(func.count(DBRerankerModel.id))) > 0
    return OptionalConfigStatus(
        langfuse=langfuse,
        default_reranker=default_reranker,
    )


def check_rag_config_need_migration(session: Session) -> NeedMigrationStatus:
    """
    Check if any configuration needs to be migrated.
    """
    chat_engines_without_kb_configured = session.exec(
        select(ChatEngine.id)
        .where(ChatEngine.deleted_at == None)
        .where(
            text(
                "JSON_EXTRACT(engine_options, '$.knowledge_base.linked_knowledge_base') IS NULL"
            )
        )
    )

    return NeedMigrationStatus(
        chat_engines_without_kb_configured=chat_engines_without_kb_configured,
    )


def remove_chat_message_recommend_questions(
    db_session: Session,
    chat_message_id: int,
) -> None:
    delete_stmt = delete(RecommendQuestion).where(
        RecommendQuestion.chat_message_id == chat_message_id
    )
    db_session.exec(delete_stmt)
    db_session.commit()


def get_chat_message_recommend_questions(
    db_session: Session,
    chat_message: DBChatMessage,
    engine_name: str = "default",
) -> List[str]:
    chat_engine_config = ChatEngineConfig.load_from_db(db_session, engine_name)
    llm = chat_engine_config.get_llama_llm(db_session)

    statement = (
        select(RecommendQuestion.questions)
        .where(RecommendQuestion.chat_message_id == chat_message.id)
        .with_for_update()  # using write lock in case the same chat message trigger multiple requests
    )

    questions = db_session.exec(statement).first()
    if questions is not None:
        return questions

    recommend_questions = llm.predict(
        prompt=get_prompt_by_jinja2_template(
            chat_engine_config.llm.further_questions_prompt,
            chat_message_content=chat_message.content,
        ),
    )
    recommend_question_list = recommend_questions.splitlines()
    recommend_question_list = [question.strip() for question in recommend_question_list if question.strip()]

    longest_question = 0
    for question in recommend_question_list:
        longest_question = max(longest_question, len(question))

    # check the output by if the output with format and the length
    if "##" in recommend_questions or "**" in recommend_questions or longest_question > 500:
        regenerate_content = f"""
        Please note that you are generating a question list. You previously generated it incorrectly; try again.
        ----------------------------------------
        {chat_message.content}
        """
        # with format or too long for per question, it's not a question list, generate again
        recommend_questions = llm.predict(
            prompt=get_prompt_by_jinja2_template(
                chat_engine_config.llm.further_questions_prompt,
                chat_message_content=regenerate_content,
            ),
        )

    db_session.add(
        RecommendQuestion(
            chat_message_id=chat_message.id,
            questions=recommend_question_list,
        )
    )
    db_session.commit()

    return recommend_question_list
