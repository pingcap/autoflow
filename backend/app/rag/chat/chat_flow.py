import json
import logging
from datetime import datetime, UTC
from typing import List, Optional, Generator, Tuple
from urllib.parse import urljoin
from uuid import UUID

import requests
from langfuse import Langfuse
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import get_response_synthesizer
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager, EventPayload
from llama_index.core.schema import NodeWithScore
from sqlmodel import Session

from app.core.config import settings
from app.exceptions import ChatNotFound
from app.models import (
    User,
    Chat as DBChat,
    ChatVisibility,
    ChatMessage as DBChatMessage,
)
from app.rag.chat.config import ChatEngineConfig
from app.rag.chat.retrieve.retrieve_flow import SourceDocument, RetrieveFlow
from app.rag.chat.stream_protocol import (
    ChatEvent,
    ChatStreamDataPayload,
    ChatStreamMessagePayload,
)
from app.rag.retrievers.knowledge_graph.schema import KnowledgeGraphRetrievalResult
from app.rag.types import ChatEventType, MessageRole, ChatMessageSate, MyCBEventType
from app.rag.utils import parse_goal_response_format
from app.repositories import chat_repo
from app.site_settings import SiteSetting
from app.utils.jinja2 import get_prompt_by_jinja2_template


logger = logging.getLogger(__name__)


def parse_chat_messages(
    chat_messages: List[ChatMessage],
) -> tuple[str, List[ChatMessage]]:
    user_question = chat_messages[-1].content
    chat_history = chat_messages[:-1]
    return user_question, chat_history


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
        self.user_question, self.chat_history = parse_chat_messages(chat_messages)
        if chat_id:
            # FIXME:
            #   only chat owner or superuser can access the chat,
            #   anonymous user can only access anonymous chat by track_id
            self.db_chat_obj = chat_repo.get(self.db_session, chat_id)
            if not self.db_chat_obj:
                raise ChatNotFound(chat_id)
            try:
                self.engine_config = ChatEngineConfig.load_from_db(
                    db_session, self.db_chat_obj.engine.name
                )
                self.db_chat_engine = self.engine_config.get_db_chat_engine()
            except Exception as e:
                logger.error(f"Failed to load chat engine config: {e}")
                self.engine_config = ChatEngineConfig.load_from_db(
                    db_session, engine_name
                )
                self.db_chat_engine = self.engine_config.get_db_chat_engine()
            logger.info(
                f"ChatService - chat_id: {chat_id}, chat_engine: {self.db_chat_obj.engine.name}"
            )
            self.chat_history = [
                ChatMessage(role=m.role, content=m.content, additional_kwargs={})
                for m in chat_repo.get_messages(self.db_session, self.db_chat_obj)
            ]
        else:
            self.engine_config = ChatEngineConfig.load_from_db(db_session, engine_name)
            self.db_chat_engine = self.engine_config.get_db_chat_engine()
            self.db_chat_obj = chat_repo.create(
                self.db_session,
                DBChat(
                    title=self.user_question[:100],
                    engine_id=self.db_chat_engine.id,
                    engine_options=self.engine_config.screenshot(),
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
        self._llm = self.engine_config.get_llama_llm(self.db_session)
        self._llm.callback_manager = self.callback_manager
        self._fast_llm = self.engine_config.get_fast_llama_llm(self.db_session)
        self._fast_llm.callback_manager = self.callback_manager
        self._fast_dspy_lm = self.engine_config.get_fast_dspy_lm(self.db_session)

        # Load knowledge bases.
        self.knowledge_bases = self.engine_config.get_knowledge_bases(self.db_session)
        self.knowledge_base_ids = [kb.id for kb in self.knowledge_bases]

        # Init retrieve flow.
        self.retrieve_flow = RetrieveFlow(
            db_session=self.db_session,
            engine_name=self.engine_name,
            engine_config=self.engine_config,
            llm=self._llm,
            fast_llm=self._fast_llm,
            knowledge_bases=self.knowledge_bases,
            callback_manager=self.callback_manager,
        )

    def _create_root_observation(self, langfuse: Langfuse):
        return langfuse.trace(
            name="chat",
            user_id=self.user.email if self.user else f"anonymous-{self.browser_id}",
            metadata={
                "chat_engine_config": self.engine_config.screenshot(),
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
                self.engine_config.external_engine_config
                and self.engine_config.external_engine_config.stream_chat_api_url
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
        (
            knowledge_graph,
            knowledge_graph_context,
        ) = yield from self._search_knowledge_graph(user_question=self.user_question)

        # 2. Refine the user question using knowledge graph and chat history.
        refined_question = yield from self._refine_user_question(
            user_question=self.user_question,
            chat_history=self.chat_history,
            knowledge_graph_context=knowledge_graph_context,
            refined_question_prompt=self.engine_config.llm.condense_question_prompt,
        )

        # 3. Check if the question provided enough context information or need to clarify.
        if self.engine_config.clarify_question:
            need_clarify, need_clarify_response = yield from self._clarify_question(
                user_question=refined_question,
                chat_history=self.chat_history,
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

        # 4. Use refined question to search for relevant chunks.
        relevant_chunks = yield from self._search_relevance_chunks(
            user_question=refined_question
        )

        # 5. Generate a response using the refined question and related chunks
        response_text, source_documents = yield from self._generate_answer(
            refined_question=refined_question,
            knowledge_graph_context=knowledge_graph_context,
            relevant_chunks=relevant_chunks,
        )

        yield from self._chat_finish(
            db_assistant_message=db_assistant_message,
            db_user_message=db_user_message,
            response_text=response_text,
            knowledge_graph=knowledge_graph,
            source_documents=source_documents,
        )

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
        return db_user_message, db_assistant_message

    def _search_knowledge_graph(
        self,
        user_question: str,
        annotation_silent: bool = False,
    ) -> Generator[ChatEvent, None, Tuple[KnowledgeGraphRetrievalResult, str]]:
        knowledge_graph = KnowledgeGraphRetrievalResult()
        knowledge_graph_context = ""
        kg_config = self.engine_config.knowledge_graph
        if kg_config is not None and kg_config.enabled:
            if not annotation_silent:
                if kg_config.using_intent_search:
                    yield ChatEvent(
                        event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                        payload=ChatStreamMessagePayload(
                            state=ChatMessageSate.KG_RETRIEVAL,
                            display="Identifying The Question's Intents and Perform Knowledge Graph Search",
                        ),
                    )
                else:
                    yield ChatEvent(
                        event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                        payload=ChatStreamMessagePayload(
                            state=ChatMessageSate.KG_RETRIEVAL,
                            display="Searching the Knowledge Graph for Relevant Context",
                        ),
                    )

            with self.callback_manager.event(
                MyCBEventType.RETRIEVE_FROM_GRAPH,
                payload={EventPayload.QUERY_STR: user_question},
            ) as event:
                knowledge_graph, knowledge_graph_context = (
                    self.retrieve_flow.search_knowledge_graph(user_question)
                )
                event.on_end(payload={EventPayload: knowledge_graph})
        return knowledge_graph, knowledge_graph_context

    def _refine_user_question(
        self,
        user_question: str,
        chat_history: Optional[List[ChatMessage]] = list,
        refined_question_prompt: Optional[str] = None,
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
            payload={
                EventPayload.QUERY_STR: user_question,
                "knowledge_graph_context": knowledge_graph_context,
            },
        ) as event:
            refined_question = self._fast_llm.predict(
                get_prompt_by_jinja2_template(
                    refined_question_prompt,
                    graph_knowledges=knowledge_graph_context,
                    chat_history=chat_history,
                    question=user_question,
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
        user_question: str,
        chat_history: Optional[List[ChatMessage]] = list,
        knowledge_graph_context: str = "",
    ) -> Generator[ChatEvent, None, Tuple[bool, str]]:
        """
        Check if the question clear and provided enough context information, otherwise, it is necessary to
        stop the conversation early and ask the user for the further clarification.

        Args:
            user_question: str
            knowledge_graph_context: str

        Returns:
            bool: Determine whether further clarification of the issue is needed from the user.
            str: The content of the questions that require clarification from the user.
        """
        with self.callback_manager.event(
            MyCBEventType.CLARIFYING_QUESTION,
            payload={EventPayload.QUERY_STR: user_question},
        ) as event:
            clarity_result = (
                self._fast_llm.predict(
                    prompt=get_prompt_by_jinja2_template(
                        self.engine_config.llm.clarifying_question_prompt,
                        graph_knowledges=knowledge_graph_context,
                        chat_history=chat_history,
                        question=user_question,
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

    def _search_relevance_chunks(
        self, user_question: str
    ) -> Generator[ChatEvent, None, List[NodeWithScore]]:
        yield ChatEvent(
            event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
            payload=ChatStreamMessagePayload(
                state=ChatMessageSate.SEARCH_RELATED_DOCUMENTS,
                display="Retrieving the Most Relevant Documents",
            ),
        )

        with self.callback_manager.event(
            MyCBEventType.RETRIEVE, payload={EventPayload.QUERY_STR: user_question}
        ) as event:
            relevant_chunks = self.retrieve_flow.search_relevant_chunks(user_question)
            event.on_end(
                payload={
                    EventPayload.CHUNKS: relevant_chunks,
                }
            )
        return relevant_chunks

    def _generate_answer(
        self,
        refined_question: str,
        knowledge_graph_context: str,
        relevant_chunks: List[NodeWithScore],
    ) -> Generator[ChatEvent, None, Tuple[str, List[SourceDocument]]]:
        # Initialize response synthesizer.
        text_qa_template = get_prompt_by_jinja2_template(
            self.engine_config.llm.text_qa_prompt,
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

        # Initialize response.
        response = response_synthesizer.synthesize(
            query=refined_question,
            nodes=relevant_chunks,
        )
        source_documents = self.retrieve_flow.get_source_documents_from_nodes(
            response.source_nodes
        )
        yield ChatEvent(
            event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
            payload=ChatStreamMessagePayload(
                state=ChatMessageSate.SOURCE_NODES,
                context=source_documents,
            ),
        )

        # Generate response.
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

    def _post_verification(
        self, user_question: str, response_text: str, chat_id: UUID, message_id: int
    ) -> Optional[str]:
        # post verification to external service, will return the post verification result url
        post_verification_url = self.engine_config.post_verification_url
        post_verification_token = self.engine_config.post_verification_token

        if not post_verification_url:
            return None

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
            return None

    def _chat_finish(
        self,
        db_assistant_message: ChatMessage,
        db_user_message: ChatMessage,
        response_text: str,
        knowledge_graph: KnowledgeGraphRetrievalResult = KnowledgeGraphRetrievalResult(),
        source_documents: Optional[List[SourceDocument]] = list,
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

        db_assistant_message.sources = [s.model_dump() for s in source_documents]
        db_assistant_message.graph_data = knowledge_graph.to_stored_graph_dict()
        db_assistant_message.content = response_text
        db_assistant_message.post_verification_result_url = post_verification_result_url
        db_assistant_message.updated_at = datetime.now(UTC)
        db_assistant_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_assistant_message)

        db_user_message.graph_data = knowledge_graph.to_stored_graph_dict()
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

    # TODO: Separate _external_chat() method into another ExternalChatFlow class, but at the same time, we need to
    #  share some common methods through ChatMixin or BaseChatFlow.
    def _external_chat(self) -> Generator[ChatEvent | str, None, None]:
        db_user_message, db_assistant_message = yield from self._chat_start()

        goal, response_format = self.user_question, {}
        try:
            # 1. Generate the goal with the user question, knowledge graph and chat history.
            goal, response_format = yield from self._generate_goal()

            # 2. Check if the goal provided enough context information or need to clarify.
            if self.engine_config.clarify_question:
                need_clarify, need_clarify_response = yield from self._clarify_question(
                    user_question=goal, chat_history=self.chat_history
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
            self.engine_config.external_engine_config.stream_chat_api_url
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
                user_question=self.user_question,
                chat_history=self.chat_history,
                refined_question_prompt=self.engine_config.llm.generate_goal_prompt,
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
