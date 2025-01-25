import json
import time
import logging
import requests

from uuid import UUID
from typing import List, Generator, Optional, Tuple
from datetime import datetime, UTC
from urllib.parse import urljoin
from llama_index.core.schema import NodeWithScore
from sqlalchemy import text, delete
from sqlmodel import Session, select, func
from llama_index.core import QueryBundle
from llama_index.core.base.llms.base import ChatMessage
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
    Entity as DBEntity,
    Relationship as DBRelationship,
    ChatEngine,
)
from app.core.config import settings
from app.models.entity import get_kb_entity_model
from app.models.recommend_question import RecommendQuestion
from app.rag.chat_stream_protocol import (
    ChatStreamMessagePayload,
    ChatStreamDataPayload,
    ChatEvent,
)
from app.models.relationship import get_kb_relationship_model
from app.rag.indices.knowledge_graph.graph_store import TiDBGraphStore
from app.rag.retrievers.knowledge_graph.fusion_retriever import (
    KnowledgeGraphFusionRetriever,
)
from app.rag.retrievers.knowledge_graph.schema import (
    KnowledgeGraphRetrieverConfig,
    KnowledgeGraphRetrievalResult,
)
from app.rag.retrievers.chunk.fusion_retriever import (
    ChunkFusionRetriever,
)
from app.rag.retrievers.chunk.schema import (
    VectorSearchRetrieverConfig,
)

from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.knowledge_base.index_store import get_kb_tidb_graph_editor
from app.rag.indices.knowledge_graph.graph_store import (
    legacy_tidb_graph_editor,
    TiDBGraphEditor,
)
from app.rag.knowledge_base.selector import KBSelectMode
from app.rag.utils import parse_goal_response_format
from app.rag.chat_config import (
    ChatEngineConfig,
    KnowledgeGraphOption,
    KnowledgeBaseOption,
)
from app.rag.embeddings.resolver import (
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
from app.utils.jinja2 import get_prompt_by_jinja2_template

logger = logging.getLogger(__name__)


class ChatFlow:
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
        self.chat_id = chat_id
        self.db_session = db_session
        self.user = user
        self.browser_id = browser_id
        self.engine_name = engine_name

        # Load chat engine and chat session.
        self.user_question, self.chat_history = self._parse_chat_messages(chat_messages)
        if chat_id:
            # FIXME:
            #   only chat owner or superuser can access the chat,
            #   anonymous user can only access anonymous chat by track_id
            self.db_chat_obj = chat_repo.get(self.db_session, chat_id)
            if not self.db_chat_obj:
                raise ChatNotFound(chat_id)
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
                    visibility=ChatVisibility.PUBLIC
                    if not self.user
                    else ChatVisibility.PRIVATE,
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

        # Init Langfuse for tracing.
        enable_langfuse = (
            SiteSetting.langfuse_secret_key and SiteSetting.langfuse_public_key
        )
        if enable_langfuse:
            # Move to global scope.
            langfuse = Langfuse(
                host=SiteSetting.langfuse_host,
                secret_key=SiteSetting.langfuse_secret_key,
                public_key=SiteSetting.langfuse_public_key,
            )
            # Why we don't use high-level decorator `observe()` as \
            #   `https://langfuse.com/docs/integrations/llama-index/get-started` suggested?
            # track:
            #   - https://github.com/langfuse/langfuse/issues/2015
            #   - https://langfuse.com/blog/2024-04-python-decorator
            root_observation = self._create_root_observation(langfuse)
            langfuse_handler = LlamaIndexCallbackHandler()
            langfuse_handler.set_root(root_observation)
            self.callback_manager = CallbackManager([langfuse_handler])
            self.trace_id = root_observation.trace_id
            self.trace_url = root_observation.get_trace_url()
        else:
            self.callback_manager = CallbackManager([])
            self.trace_id = None
            self.trace_url = ""

        # Init LLM.
        self._llm = self.chat_engine_config.get_llama_llm(self.db_session)
        self._llm.callback_manager = self.callback_manager
        self._fast_llm = self.chat_engine_config.get_fast_llama_llm(self.db_session)
        self._fast_llm.callback_manager = self.callback_manager
        self._fast_dspy_lm = self.chat_engine_config.get_fast_dspy_lm(self.db_session)

        # Load knowledge bases.
        kb_config: KnowledgeBaseOption = self.chat_engine_config.knowledge_base
        linked_knowledge_base_ids = []
        if len(kb_config.linked_knowledge_bases) == 0:
            linked_knowledge_base_ids.append(
                self.chat_engine_config.knowledge_base.linked_knowledge_base.id
            )
        else:
            linked_knowledge_base_ids.extend(
                [kb.id for kb in kb_config.linked_knowledge_bases]
            )
        self.knowledge_base_ids = linked_knowledge_base_ids
        self.knowledge_bases = knowledge_base_repo.get_by_ids(
            self.db_session, knowledge_base_ids=linked_knowledge_base_ids
        )

    def _create_root_observation(self, langfuse: Langfuse):
        return langfuse.trace(
            name="chat",
            user_id=self.user.email if self.user else f"anonymous-{self.browser_id}",
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

    def chat(self) -> Generator[ChatEvent | str, None, None]:
        try:
            self.callback_manager.start_trace(self.trace_id)
            if (
                self.chat_engine_config.external_engine_config
                and self.chat_engine_config.external_engine_config.stream_chat_api_url
            ):
                yield from self._external_chat()
            else:
                yield from self._builtin_chat()
            self.callback_manager.end_trace(self.trace_id)
        except Exception as e:
            self.callback_manager.end_trace(self.trace_id)
            logger.exception(e)
            yield ChatEvent(
                event_type=ChatEventType.ERROR_PART,
                payload="Encountered an error while processing the chat. Please try again later.",
            )

    def _builtin_chat(self) -> Generator[ChatEvent | str, None, None]:
        db_user_message, db_assistant_message = yield from self._chat_start()

        # 1. Retrieve Knowledge graph related to the user question.
        kg_config = self.chat_engine_config.knowledge_graph
        knowledge_graph = KnowledgeGraphRetrievalResult()
        knowledge_graph_context = ""
        if kg_config is not None and kg_config.enabled:
            (
                knowledge_graph,
                knowledge_graph_context,
            ) = yield from self._search_knowledge_graph(kg_config=kg_config)

        # 2. Refine the user question using knowledge graph and chat history.
        refined_question = yield from self._refine_user_question(
            knowledge_graph_context=knowledge_graph_context,
            refined_question_prompt=self.chat_engine_config.llm.condense_question_prompt,
            annotation_silent=False,
        )

        # 3. Check if the question provided enough context information or need to clarify.
        if self.chat_engine_config.clarify_question:
            need_clarify, need_clarify_response = yield from self._clarify_question(
                refined_question=refined_question,
                knowledge_graph_context=knowledge_graph_context,
            )
            if need_clarify:
                yield from self._chat_finish(
                    db_assistant_message=db_assistant_message,
                    db_user_message=db_user_message,
                    response_text=need_clarify_response,
                    knowledge_graph=knowledge_graph,
                )
                return

        # 4. Use refined questions to search for relevant chunks.
        relevant_chunk_nodes = yield from self._search_relevant_chunks(
            refined_question=refined_question
        )

        # 5. Generate a response using the refined question and related chunks
        response_text, source_documents = yield from self._generate_answer(
            refined_question=refined_question,
            knowledge_graph_context=knowledge_graph_context,
            relevant_chunk_nodes=relevant_chunk_nodes,
        )

        yield from self._chat_finish(
            db_assistant_message=db_assistant_message,
            db_user_message=db_user_message,
            response_text=response_text,
            knowledge_graph=knowledge_graph,
            source_documents=source_documents,
        )

    # TODO: Separate _external_chat() method into another ExternalChatFlow class, but at the same time, we need to
    #  share some common methods through ChatMixin or BaseChatFlow.
    def _external_chat(self) -> Generator[ChatEvent | str, None, None]:
        db_user_message, db_assistant_message = yield from self._chat_start()

        goal, response_format = self.user_question, {}
        try:
            # 1. Generate the goal with the user question, knowledge graph and chat history.
            goal, response_format = yield from self._generate_goal()

            # 2. Check if the goal provided enough context information or need to clarify.
            if self.chat_engine_config.clarify_question:
                need_clarify, need_clarify_response = yield from self._clarify_question(
                    refined_question=goal,
                )
                if need_clarify:
                    logger.info("", extra={"chat_id": self.chat_id})
                    yield from self._chat_finish(
                        db_assistant_message=db_assistant_message,
                        db_user_message=db_user_message,
                        response_text=need_clarify_response,
                        annotation_silent=True,
                    )
                    return
        except Exception as e:
            goal = self.user_question
            logger.warning(
                f"Failed to generate refined goal, fallback to use user question as goal directly: {e}",
                exc_info=True,
                extra={},
            )

        cache_messages = None
        if settings.ENABLE_QUESTION_CACHE:
            try:
                logger.info(
                    f"start to find_recent_assistant_messages_by_goal with goal: {goal}, response_format: {response_format}"
                )
                cache_messages = chat_repo.find_recent_assistant_messages_by_goal(
                    self.db_session,
                    {"goal": goal, "Lang": response_format.get("Lang", "English")},
                    90,
                )
                logger.info(
                    f"find_recent_assistant_messages_by_goal result {len(cache_messages)} for goal {goal}"
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
                "response_format": response_format,
                "namespace_name": "Default",
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
        try:
            post_verification_result_url = self._post_verification(
                goal,
                response_text,
                self.db_chat_obj.id,
                db_assistant_message.id,
            )
            db_assistant_message.post_verification_result_url = (
                post_verification_result_url
            )
        except Exception:
            logger.error(
                "Specific error occurred during post verification job.", exc_info=True
            )

        trace_url = f"{base_url}?task_id={task_id}" if task_id else ""
        message_meta = {
            "task_id": task_id,
            "goal": goal,
            **response_format,
        }

        db_assistant_message.content = response_text
        db_assistant_message.trace_url = trace_url
        db_assistant_message.meta = message_meta
        db_assistant_message.updated_at = datetime.now(UTC)
        db_assistant_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_assistant_message)

        db_user_message.trace_url = trace_url
        db_user_message.meta = message_meta
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

    def _generate_goal(self) -> Generator[ChatEvent, None, Tuple[str, dict]]:
        try:
            refined_question = yield from self._refine_user_question(
                refined_question_prompt=self.chat_engine_config.llm.generate_goal_prompt,
                annotation_silent=True,
            )

            goal = refined_question.strip()
            if goal.startswith("Goal: "):
                goal = goal[len("Goal: ") :].strip()
        except Exception as e:
            logger.error(f"Failed to refine question with related knowledge graph: {e}")
            goal = self.user_question

        response_format = {}
        try:
            clean_goal, response_format = parse_goal_response_format(goal)
            logger.info(f"clean goal: {clean_goal}, response_format: {response_format}")
            if clean_goal:
                goal = clean_goal
        except Exception as e:
            logger.error(f"Failed to parse goal and response format: {e}")

        return goal, response_format

    def _chat_start(
        self,
    ) -> Generator[ChatEvent, None, Tuple[DBChatMessage, DBChatMessage]]:
        db_user_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.USER.value,
                trace_url=self.trace_url,
                content=self.user_question,
            ),
        )
        db_assistant_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.ASSISTANT.value,
                trace_url=self.trace_url,
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
        yield ChatEvent(
            event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
            payload=ChatStreamMessagePayload(
                state=ChatMessageSate.TRACE,
                context={"langfuse_url": self.trace_url},
            ),
        )
        return db_user_message, db_assistant_message

    def _search_knowledge_graph(
        self,
        kg_config: KnowledgeGraphOption,
        annotation_silent: bool = False,
    ) -> Generator[ChatEvent, None, Tuple[KnowledgeGraphRetrievalResult, str]]:
        """
        Search the knowledge graph for relevant entities, relationships, and chunks.
        Args:
            kg_config: KnowledgeGraphOption
            annotation_silent: bool, if True, do not send annotation events

        Returns:
            KnowledgeGraphRetrievalResult: The retrieved knowledge graph.
            str: graph_knowledges_context
        """
        # For forward compatibility of chat engine config.
        enable_metadata_filter = kg_config.enable_metadata_filter or (
            kg_config.relationship_meta_filters is not None
        )
        metadata_filters = (
            kg_config.metadata_filters or kg_config.relationship_meta_filters
        )

        kg_retriever = KnowledgeGraphFusionRetriever(
            db_session=self.db_session,
            knowledge_base_ids=self.knowledge_base_ids,
            llm=self._llm,
            use_query_decompose=kg_config.using_intent_search,
            use_async=True,
            kb_select_mode=KBSelectMode.SINGLE_SECTION,
            config=KnowledgeGraphRetrieverConfig(
                depth=kg_config.depth,
                include_metadata=kg_config.include_meta,
                with_degree=kg_config.with_degree,
                enable_metadata_filter=enable_metadata_filter,
                metadata_filters=metadata_filters,
            ),
            callback_manager=self.callback_manager,
        )

        if kg_config.using_intent_search:
            if not annotation_silent:
                yield ChatEvent(
                    event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                    payload=ChatStreamMessagePayload(
                        state=ChatMessageSate.KG_RETRIEVAL,
                        display="Identifying The Question's Intents and Perform Knowledge Graph Search",
                    ),
                )

            with self.callback_manager.event(
                MyCBEventType.GRAPH_SEMANTIC_SEARCH,
                payload={EventPayload.QUERY_STR: self.user_question},
            ) as event:
                knowledge_graph = kg_retriever.retrieve_knowledge_graph(
                    self.user_question
                )
                kg_context_template = get_prompt_by_jinja2_template(
                    self.chat_engine_config.llm.intent_graph_knowledge,
                    # For forward compatibility considerations.
                    sub_queries=knowledge_graph.to_subqueries_dict(),
                )
                knowledge_graph_context = kg_context_template.template

                event.on_end(
                    payload={
                        "knowledge_graph": knowledge_graph,
                        "knowledge_graph_context": knowledge_graph_context,
                    }
                )
        else:
            if not annotation_silent:
                yield ChatEvent(
                    event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                    payload=ChatStreamMessagePayload(
                        state=ChatMessageSate.KG_RETRIEVAL,
                        display="Searching the Knowledge Graph for Relevant Context",
                    ),
                )

            with self.callback_manager.event(
                MyCBEventType.RETRIEVE_FROM_GRAPH,
                payload={EventPayload.QUERY_STR: self.user_question},
            ) as event:
                knowledge_graph = kg_retriever.retrieve_knowledge_graph(
                    self.user_question
                )
                kg_context_template = get_prompt_by_jinja2_template(
                    self.chat_engine_config.llm.normal_graph_knowledge,
                    entities=knowledge_graph.entities,
                    relationships=knowledge_graph.relationships,
                )
                knowledge_graph_context = kg_context_template.template

                event.on_end(
                    payload={
                        "knowledge_graph": knowledge_graph,
                        "knowledge_graph_context": knowledge_graph_context,
                    }
                )

        return (
            knowledge_graph,
            knowledge_graph_context,
        )

    def _refine_user_question(
        self,
        refined_question_prompt: str,
        knowledge_graph_context: str = "",
        annotation_silent: bool = False,
    ) -> Generator[ChatEvent, None, str]:
        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.REFINE_QUESTION,
                    display="Query Rewriting for Enhanced Information Retrieval",
                ),
            )

        with self.callback_manager.event(
            MyCBEventType.REFINE_QUESTION,
            payload={EventPayload.QUERY_STR: self.user_question},
        ) as event:
            refined_question = self._fast_llm.predict(
                get_prompt_by_jinja2_template(
                    refined_question_prompt,
                    graph_knowledges=knowledge_graph_context,
                    chat_history=self.chat_history,
                    question=self.user_question,
                    current_date=datetime.now().strftime("%Y-%m-%d"),
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
        return refined_question

    def _clarify_question(
        self,
        refined_question: str,
        knowledge_graph_context: str = "",
    ) -> Generator[ChatEvent, None, Tuple[bool, str]]:
        """
        Check if the question clear and provided enough context information, otherwise, it is necessary to
        stop the conversation early and ask the user for the further clarification.

        Args:
            refined_question: str
            knowledge_graph_context: str

        Returns:
            bool: Determine whether further clarification of the issue is needed from the user.
            str: The content of the questions that require clarification from the user.
        """
        with self.callback_manager.event(
            MyCBEventType.CLARIFYING_QUESTION,
            payload={EventPayload.QUERY_STR: refined_question},
        ) as event:
            clarity_result = (
                self._fast_llm.predict(
                    prompt=get_prompt_by_jinja2_template(
                        self.chat_engine_config.llm.clarifying_question_prompt,
                        graph_knowledges=knowledge_graph_context,
                        chat_history=self.chat_history,
                        question=refined_question,
                    ),
                )
                .strip()
                .strip(".\"'!")
            )

            need_clarify = clarity_result.lower() != "false"
            need_clarify_response = clarity_result if need_clarify else ""
            event.on_end(
                payload={
                    "need_clarify": need_clarify,
                    "need_clarify_response": need_clarify_response,
                }
            )

        if need_clarify:
            yield ChatEvent(
                event_type=ChatEventType.TEXT_PART,
                payload=need_clarify_response,
            )

        return need_clarify, need_clarify_response

    def _search_relevant_chunks(
        self, refined_question: str, annotation_silent: bool = False
    ) -> Generator[ChatEvent, None, List[NodeWithScore]]:
        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.SEARCH_RELATED_DOCUMENTS,
                    display="Retrieving the Most Relevant Documents",
                ),
            )

        with self.callback_manager.event(
            MyCBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: refined_question},
        ) as event:
            retriever = ChunkFusionRetriever(
                db_session=self.db_session,
                knowledge_base_ids=self.knowledge_base_ids,
                llm=self._llm,
                config=VectorSearchRetrieverConfig(
                    similarity_top_k=60,
                    oversampling_factor=5,
                    top_k=10,
                ),
                use_query_decompose=False,
                use_async=True,
                callback_manager=self.callback_manager,
            )

            nodes_with_score = retriever.retrieve(QueryBundle(refined_question))

            event.on_end(
                payload={
                    "nodes_with_score": nodes_with_score,
                }
            )

        return nodes_with_score

    def _generate_answer(
        self,
        refined_question: str,
        knowledge_graph_context: str,
        relevant_chunk_nodes: List[NodeWithScore],
        annotation_silent: bool = False,
    ) -> Generator[ChatEvent, None, Tuple[str, List[dict]]]:
        text_qa_template = get_prompt_by_jinja2_template(
            self.chat_engine_config.llm.text_qa_prompt,
            current_date=datetime.now().strftime("%Y-%m-%d"),
            graph_knowledges=knowledge_graph_context,
            original_question=self.user_question,
        )
        response_synthesizer = get_response_synthesizer(
            llm=self._llm,
            text_qa_template=text_qa_template,
            streaming=True,
            callback_manager=self.callback_manager,
        )
        response = response_synthesizer.synthesize(
            query=refined_question, nodes=relevant_chunk_nodes
        )
        source_documents = self._get_source_documents(response)

        yield ChatEvent(
            event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
            payload=ChatStreamMessagePayload(
                state=ChatMessageSate.SOURCE_NODES,
                context=source_documents,
            ),
        )
        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.GENERATE_ANSWER,
                    display="Generating a Precise Answer with AI",
                ),
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

        return response_text, source_documents

    def _parse_chat_messages(
        self, chat_messages: List[ChatMessage]
    ) -> tuple[str, List[ChatMessage]]:
        user_question = chat_messages[-1].content
        chat_history = chat_messages[:-1]
        return user_question, chat_history

    def _get_source_documents(self, response: StreamingResponse) -> List[dict]:
        document_ids = [
            s_n.node.metadata["document_id"] for s_n in response.source_nodes
        ]
        stmt = select(DBDocument.id, DBDocument.name, DBDocument.source_uri).where(
            DBDocument.id.in_(document_ids)
        )
        rows = self.db_session.exec(stmt).all()

        # Keep the original order of document ids, which is sorted by similarity.
        rows = sorted(rows, key=lambda row: document_ids.index(row[0]))
        return [{"id": row[0], "name": row[1], "source_uri": row[2]} for row in rows]

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

    def _chat_finish(
        self,
        db_assistant_message: ChatMessage,
        db_user_message: ChatMessage,
        response_text: str,
        knowledge_graph: KnowledgeGraphRetrievalResult = KnowledgeGraphRetrievalResult(),
        source_documents: Optional[List[dict]] = list,
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

        graph_data = knowledge_graph.to_graph_data_dict()

        db_assistant_message.sources = source_documents
        db_assistant_message.graph_data = graph_data
        db_assistant_message.content = response_text
        db_assistant_message.post_verification_result_url = post_verification_result_url
        db_assistant_message.updated_at = datetime.now(UTC)
        db_assistant_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_assistant_message)

        db_user_message.graph_data = graph_data
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


def user_can_view_chat(chat: DBChat, user: Optional[User]) -> bool:
    # Anonymous or public chat can be accessed by anyone
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
    entities, relations = graph_store.retrieve_knowledge_graph(
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
    recommend_question_list = [
        question.strip() for question in recommend_question_list if question.strip()
    ]

    longest_question = 0
    for question in recommend_question_list:
        longest_question = max(longest_question, len(question))

    # check the output by if the output with format and the length
    if (
        "##" in recommend_questions
        or "**" in recommend_questions
        or longest_question > 500
    ):
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
