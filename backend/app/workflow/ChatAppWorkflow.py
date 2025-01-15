from datetime import datetime
import logging
from typing import List, Optional, Type
from uuid import UUID

import dspy
from llama_index.core import get_response_synthesizer
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager, EventPayload
from llama_index.core.llms import LLM
from llama_index.core.prompts.mixin import PromptMixin
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    Event,
    StopEvent,
)
from numpy.distutils.command.config import config
from sqlmodel import Session, select, SQLModel

from app.models.chunk import get_kb_chunk_model
from app.rag.chat import get_prompt_by_jinja2_template
from app.rag.chat_config import ChatEngineConfig
from app.models import (
    Chat as DBChat,
    ChatEngine as DBChatEngine,
    Document as DBDocument,
)
from app.rag.types import MyCBEventType
from app.site_settings import SiteSetting
from langfuse import Langfuse
from app.rag.knowledge_graph.base import KnowledgeGraphIndex
from app.rag.knowledge_graph.graph_store.tidb_graph_store import TiDBGraphStore
from app.models.entity import get_kb_entity_model
from app.repositories import knowledge_base_repo
from app.models.knowledge_base import KnowledgeBase
from app.models.relationship import get_kb_relationship_model
from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.retrievers.vector_search.retriever import KBVectorSearchRetriever
from app.rag.retrievers.vector_search.config import VectorSearchConfig

logger = logging.getLogger(__name__)


class AppChatEngine:
    workflow: Workflow

    def __init__(self, db_engine: DBChatEngine):
        self.config = config
        self.workflow = Workflow()

    async def chat(
        self,
        *,
        db_chat: DBChat,
        user_question: Optional[UUID] = None,
        chat_history: List[ChatMessage],
    ):
        result = await self.workflow.run(
            user_question=user_question,
            chat_history=chat_history,
        )
        return result


class SearchKnowledgeGraphEvent(Event):
    """Search knowledge graph event"""


class RefineQuestionEvent(Event):
    """Refine question event"""


class RetrieveEvent(Event):
    """Retrieve event"""


class ClarifyQuestionEvent(Event):
    """Clarify question event"""


class GenerateAnswerEvent(Event):
    """Generate answer event"""


class GenerateAnswerStreamEvent(Event):
    """Generate answer stream event"""

    def __init__(self, chunk: str):
        super().__init__()
        self.chunk = chunk


class EarlyStopEvent(Event):
    """Early stop event"""

    def __init__(self, answer: str, **kwargs):
        super().__init__(**kwargs)
        self.answer = answer


class AppChatFlow(Workflow, PromptMixin):
    db_session: Session
    config: ChatEngineConfig
    llm: LLM
    fast_llm: LLM
    fast_dspy_lm: dspy.LM
    embed_model: BaseEmbedding
    knowledge_base: KnowledgeBase
    callback_manager: CallbackManager = CallbackManager([])

    def __init__(
        self,
        db_session: Session,
        engine_name: str,
        engine_config: ChatEngineConfig,
        langfuse: Optional[Langfuse] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.db_session = db_session
        self.engine_name = engine_name
        self.config = engine_config
        self.llm = engine_config.get_llama_llm(db_session)
        self.fast_llm = engine_config.get_fast_llama_llm(db_session)
        self.fast_dspy_llm = engine_config.get_dspy_lm(db_session)
        self.langfuse = langfuse or Langfuse(
            host=SiteSetting.langfuse_host,
            secret_key=SiteSetting.langfuse_secret_key,
            public_key=SiteSetting.langfuse_public_key,
        )
        self.knowledge_base = knowledge_base_repo.must_get(
            self.db_session,
            self.config.knowledge_base.linked_knowledge_base.id,
        )
        self.embed_model = get_kb_embed_model(self.db_session, self.knowledge_base)

    @step
    async def start_chat(
        self, ctx: Context, ev: StartEvent
    ) -> SearchKnowledgeGraphEvent | RefineQuestionEvent:
        user_question = ev.get("user_question")
        chat_history = ev.get("chat_history")

        await ctx.set("user_question", user_question)
        await ctx.set("chat_history", chat_history)

        if self.config.knowledge_graph.enabled:
            return SearchKnowledgeGraphEvent(
                user_question=user_question, chat_history=chat_history
            )
        else:
            return RefineQuestionEvent(
                user_question=user_question, chat_history=chat_history
            )

    @step
    async def search_knowledge_graph(
        self, ctx: Context, ev: SearchKnowledgeGraphEvent
    ) -> RefineQuestionEvent:
        user_question = ev.get("user_question")
        chat_history = ev.get("chat_history")

        with self.callback_manager.as_trace("search_knowledge_graph"):
            with self.callback_manager.event(
                MyCBEventType.GRAPH_SEMANTIC_SEARCH,
                payload={EventPayload.QUERY_STR: user_question},
            ) as event:
                kg_config = self.config.knowledge_graph
                entity_db_model = get_kb_entity_model(self.knowledge_base)
                relationship_db_model = get_kb_relationship_model(self.knowledge_base)
                graph_store = TiDBGraphStore(
                    dspy_lm=self.fast_dspy_lm,
                    session=self.db_session,
                    embed_model=self.embed_model,
                    entity_db_model=entity_db_model,
                    relationship_db_model=relationship_db_model,
                )
                graph_index: KnowledgeGraphIndex = KnowledgeGraphIndex.from_existing(
                    dspy_lm=self.fast_dspy_lm,
                    kg_store=graph_store,
                    callback_manager=self.callback_manager,
                )

                if kg_config.using_intent_search:
                    graph_index._callback_manager = self.callback_manager
                    sub_queries = graph_index.intent_analyze(
                        user_question, chat_history
                    )
                    result = graph_index.graph_semantic_search(
                        sub_queries,
                        depth=kg_config.depth,
                        include_meta=kg_config.include_meta,
                        relationship_meta_filters=kg_config.relationship_meta_filters,
                    )

                    entities = result["graph"]["entities"]
                    relations = result["graph"]["relationships"]
                    graph_data_source_ids = {
                        "entities": [e["id"] for e in entities],
                        "relationships": [r["id"] for r in relations],
                    }
                    graph_knowledges = get_prompt_by_jinja2_template(
                        self.config.llm.intent_graph_knowledge,
                        sub_queries=result["queries"],
                    )
                    knowledge_graph_context = graph_knowledges.template
                else:
                    entities, relations, _ = graph_index.retrieve_with_weight(
                        user_question,
                        [],
                        depth=kg_config.depth,
                        include_meta=kg_config.include_meta,
                        with_degree=kg_config.with_degree,
                        with_chunks=False,
                        relationship_meta_filters=kg_config.relationship_meta_filters,
                    )
                    graph_data_source_ids = {
                        "entities": [e["id"] for e in entities],
                        "relationships": [r["id"] for r in relations],
                    }
                    graph_knowledges = get_prompt_by_jinja2_template(
                        self.config.llm.normal_graph_knowledge,
                        entities=entities,
                        relationships=relations,
                    )
                    knowledge_graph_context = graph_knowledges.template

        await ctx.set("graph_data_source_ids", graph_data_source_ids)
        await ctx.set("knowledge_graph_context", knowledge_graph_context)

        return RefineQuestionEvent(user_question=user_question)

    @step
    async def condense_question(
        self, ctx: Context, ev: RefineQuestionEvent
    ) -> ClarifyQuestionEvent | RetrieveEvent:
        user_question = ev.get("user_question")
        chat_history = ev.get("chat_history", [])
        knowledge_graph_context = ctx.get("knowledge_graph_context")

        with self.callback_manager.as_trace("trace_id"):
            with self.callback_manager.event(
                MyCBEventType.CONDENSE_QUESTION,
                payload={
                    "user_question": user_question,
                    "chat_history": chat_history,
                    "knowledge_graph_context": knowledge_graph_context,
                },
            ) as event:
                condense_question_prompt = get_prompt_by_jinja2_template(
                    self.config.llm.condense_question_prompt,
                    question=user_question,
                    chat_history=chat_history,
                    graph_knowledges=knowledge_graph_context,
                    current_date=datetime.now().strftime("%Y-%m-%d"),
                )
                refined_question = self.fast_llm.predict(condense_question_prompt)
                event.on_end(payload={"refined_question": refined_question})

        await ctx.set("refined_question", refined_question)
        if self.config.clarify_question:
            return ClarifyQuestionEvent(user_question=user_question)
        else:
            return RetrieveEvent(user_question=user_question)

    @step
    async def clarify_question(
        self, ctx: Context, ev: ClarifyQuestionEvent
    ) -> EarlyStopEvent | RetrieveEvent:
        refined_question = ctx.get("refined_question")
        chat_history = ctx.get("chat_history")
        graph_knowledges_context = ctx.get("graph_knowledges_context")

        with self.callback_manager.as_trace("clarify_question"):
            with self.callback_manager.event(
                MyCBEventType.CLARIFYING_QUESTION,
                payload={EventPayload.QUERY_STR: refined_question},
            ) as event:
                clarity_result = (
                    self.fast_llm.structured_predict(
                        prompt=get_prompt_by_jinja2_template(
                            self.config.llm.clarifying_question_prompt,
                            graph_knowledges=graph_knowledges_context,
                            chat_history=chat_history,
                            question=refined_question,
                        ),
                    )
                    .strip()
                    .strip(".\"'!")
                )

                clarity_needed = clarity_result.lower() != "false"

                event.on_end(
                    payload={
                        "clarify_needed": clarity_needed,
                        "clarify_question": clarity_result,
                    }
                )

        if clarity_needed:
            return EarlyStopEvent(
                answer=clarity_result,
            )
        else:
            return RetrieveEvent(
                question=clarity_result,
            )

    @step
    async def retrieve_relevant_chunks(self, ctx: Context, ev: RetrieveEvent):
        refined_question = await ctx.get("refined_question")

        with self.callback_manager.as_trace("retrieve_relevant_chunks"):
            with self.callback_manager.event(
                MyCBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: refined_question},
            ) as event:
                chunk_model = get_kb_chunk_model(self.knowledge_base)
                retriever = KBVectorSearchRetriever(
                    config=VectorSearchConfig(
                        knowledge_base_id=self.knowledge_base.id,
                        similarity_top_k=10,
                        oversampling_factor=5,
                        top_k=5,
                    )
                )

                nodes_with_score = retriever.retrieve()
                source_documents = self._get_source_documents(
                    chunk_model, nodes_with_score
                )

                event.on_end(
                    payload={
                        "nodes_with_score": nodes_with_score,
                        "source_documents": source_documents,
                    }
                )

        await ctx.set("nodes_with_score", nodes_with_score)
        await ctx.set("source_documents", source_documents)

        return GenerateAnswerEvent()

    def _get_source_documents(
        self, chunk_model: Type[SQLModel], nodes_with_score: List[NodeWithScore]
    ) -> List[dict]:
        source_nodes_ids = [n.node.node_id for n in nodes_with_score]
        stmt = (
            select(
                chunk_model.id,
                DBDocument.id,
                DBDocument.name,
                DBDocument.source_uri,
            )
            .outerjoin(DBDocument, chunk_model.document_id == DBDocument.id)
            .where(
                chunk_model.id.in_(source_nodes_ids),
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
            if s[1] in source_documents_ids:
                continue
            source_documents_ids.append(s[1])
            source_documents.append(
                {
                    "id": s[1],
                    "name": s[2],
                    "source_uri": s[3],
                }
            )
        return source_documents

    @step
    async def generate_answer(self, ctx: Context, ev: GenerateAnswerEvent) -> StopEvent:
        user_question = await ctx.get("user_question")
        nodes_with_score = await ctx.get("nodes_with_score")
        knowledge_graph_context = await ctx.get("knowledge_graph_context")

        with self.callback_manager.as_trace("generate_answer"):
            with self.callback_manager.event(
                MyCBEventType.SYNTHESIZE,
                payload={EventPayload.QUERY_STR: user_question},
            ) as event:
                text_qa_template = get_prompt_by_jinja2_template(
                    self.config.llm.text_qa_prompt,
                    current_date=datetime.now().strftime("%Y-%m-%d"),
                    graph_knowledges=knowledge_graph_context,
                    original_question=user_question,
                )
                refine_template = get_prompt_by_jinja2_template(
                    self.config.llm.refine_prompt,
                    graph_knowledges=knowledge_graph_context,
                    original_question=user_question,
                )
                synthesizer = get_response_synthesizer(
                    llm=self.llm,
                    text_qa_template=text_qa_template,
                    refine_template=refine_template,
                    response_mode=ResponseMode.COMPACT,
                    callback_manager=self.callback_manager,
                    streaming=True,
                )
                response = synthesizer.synthesize(user_question, nodes_with_score)

                response_text = ""
                for chunk in response.response_gen:
                    response_text += chunk
                    ctx.write_event_to_stream(GenerateAnswerStreamEvent(chunk=chunk))

                event.on_end(payload=response_text)

        return StopEvent()

    @step
    async def finish(self, ctx: Context, ev: StopEvent):
        pass
