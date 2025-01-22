from datetime import datetime
import logging

from typing import List, Optional, Type

import dspy
from fastapi.responses import StreamingResponse
from llama_index.core import get_response_synthesizer
from llama_index.core.callbacks import EventPayload, CallbackManager
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)
from llama_index.core.llms.llm import LLM
from sqlmodel import Session, select, SQLModel
from app.rag.indices.knowledge_graph.retriever.fusion_retriever import (
    KnowledgeGraphFusionRetriever,
)
from app.rag.indices.knowledge_graph.retriever.schema import (
    KnowledgeGraphRetrieverConfig,
)
from app.rag.indices.vector_search.retriever.fusion_retriever import (
    VectorSearchFusionRetriever,
)
from app.rag.indices.vector_search.retriever.schema import VectorSearchRetrieverConfig
from app.rag.knowledge_base.selector import KBSelectMode
from app.utils.jinja2 import get_prompt_by_jinja2_template
from app.rag.chat_config import ChatEngineConfig, KnowledgeBaseOption
from app.models import Document as DBDocument
from app.rag.types import MyCBEventType
from app.rag.workflows.chat_flow.events import (
    SearchKnowledgeGraphEvent,
    RefineQuestionEvent,
    ClarifyQuestionEvent,
    RetrieveEvent,
    GenerateAnswerEvent,
    GenerateAnswerStreamEvent,
)
from app.site_settings import SiteSetting
from langfuse import Langfuse
from app.repositories import knowledge_base_repo


logger = logging.getLogger(__name__)


class ChatFlow(Workflow):
    """
    AppChatFlow is a standard chatting process for document-based document robots. It includes several key steps
    such as question rewriting, knowledge retrieval, and answer generation.
    """

    # Notice: ChatFlow should be reusable in different chat sessions, for example: the configuration of chat engine.
    # If you need to add session-specific variables, please use ctx.set() / ctx.get()
    config: ChatEngineConfig
    langfuse: Optional[Langfuse] = None
    db_session: Session
    llm: LLM
    fast_llm: LLM
    dspy_fast_lm: dspy.LM
    callback_manager: CallbackManager = CallbackManager([])

    def __init__(
        self,
        db_session: Session,
        config: ChatEngineConfig,
        langfuse: Optional[Langfuse] = None,
        timeout: Optional[float] = 120.0,
        **kwargs,
    ):
        super().__init__(timeout=timeout, **kwargs)
        self.config = config
        self.langfuse = langfuse or Langfuse(
            host=SiteSetting.langfuse_host,
            secret_key=SiteSetting.langfuse_secret_key,
            public_key=SiteSetting.langfuse_public_key,
        )
        self.db_session = db_session
        self.llm = self.config.get_llama_llm(db_session)
        self.fast_llm = self.config.get_fast_llama_llm(db_session)
        self.fast_dspy_lm = self.config.get_fast_dspy_lm(db_session)
        self.callback_manager = CallbackManager([])

    @step
    async def start_chat(
        self, ctx: Context, ev: StartEvent
    ) -> SearchKnowledgeGraphEvent | RefineQuestionEvent:
        for key, value in ev.items():
            await ctx.set(key, value)

        # Linked knowledge bases.
        kb_config: KnowledgeBaseOption = self.config.knowledge_base
        linked_knowledge_base_ids = []
        if len(kb_config.linked_knowledge_bases) == 0:
            linked_knowledge_base_ids.append(
                self.config.knowledge_base.linked_knowledge_base.id
            )
        else:
            linked_knowledge_base_ids.extend(
                [kb.id for kb in kb_config.linked_knowledge_bases]
            )
        knowledge_bases = knowledge_base_repo.get_by_ids(
            self.db_session, knowledge_base_ids=linked_knowledge_base_ids
        )

        await ctx.set("knowledge_base_ids", linked_knowledge_base_ids)
        await ctx.set("knowledge_bases", knowledge_bases)

        if self.config.knowledge_graph.enabled:
            return SearchKnowledgeGraphEvent()
        else:
            return RefineQuestionEvent()

    @step
    async def search_knowledge_graph(
        self, ctx: Context, ev: SearchKnowledgeGraphEvent
    ) -> RefineQuestionEvent:
        user_question: str = await ctx.get("user_question")
        knowledge_base_ids: list[int] = await ctx.get("knowledge_base_ids")

        with self.callback_manager.as_trace("search_knowledge_graph"):
            with self.callback_manager.event(
                MyCBEventType.GRAPH_SEMANTIC_SEARCH,
                payload={EventPayload.QUERY_STR: user_question},
            ) as event:
                kg_config = self.config.knowledge_graph

                # For parameter compatibility.
                enable_metadata_filter = kg_config.enable_metadata_filter or (
                    kg_config.relationship_meta_filters is not None
                )
                metadata_filters = (
                    kg_config.metadata_filters or kg_config.relationship_meta_filters
                )

                kg_retriever = KnowledgeGraphFusionRetriever(
                    db_session=self.db_session,
                    knowledge_base_ids=knowledge_base_ids,
                    llm=self.llm,
                    use_query_decompose=kg_config.using_intent_search,
                    select_mode=KBSelectMode.SINGLE_SECTION,
                    config=KnowledgeGraphRetrieverConfig(
                        depth=kg_config.depth,
                        include_metadata=kg_config.include_meta,
                        with_degree=kg_config.with_degree,
                        enable_metadata_filter=enable_metadata_filter,
                        metadata_filters=metadata_filters,
                    ),
                )

                knowledge_graph = kg_retriever.retrieve_knowledge_graph(
                    QueryBundle(user_question)
                )

                if kg_config.using_intent_search:
                    # compatibility considerations.
                    sub_queries = {}
                    for subquery in knowledge_graph.subqueries:
                        sub_queries[subquery.query] = {
                            "entities": subquery.entities,
                            "relationships": subquery.relationships,
                        }
                    kg_context_template = get_prompt_by_jinja2_template(
                        self.config.llm.intent_graph_knowledge,
                        sub_queries=sub_queries,
                    )
                    kg_context_str = kg_context_template.template
                else:
                    kg_context_template = get_prompt_by_jinja2_template(
                        self.config.llm.normal_graph_knowledge,
                        entities=knowledge_graph.entities,
                        relationships=knowledge_graph.relationships,
                    )
                    kg_context_str = kg_context_template.template

                event.on_end(
                    payload={
                        "knowledge_graph": knowledge_graph,
                        "knowledge_graph_context_str": kg_context_str,
                    }
                )

        await ctx.set("knowledge_graph", knowledge_graph)
        await ctx.set("knowledge_graph_context_str", kg_context_str)

        return RefineQuestionEvent(user_question=user_question)

    @step
    async def refine_question(
        self, ctx: Context, ev: RefineQuestionEvent
    ) -> ClarifyQuestionEvent | RetrieveEvent:
        user_question = ctx.get("user_question")
        chat_history = ctx.get("chat_history", [])
        knowledge_graph_context_str = ctx.get("knowledge_graph_context_str")

        with self.callback_manager.as_trace("refine_question"):
            with self.callback_manager.event(
                MyCBEventType.CONDENSE_QUESTION,
                payload={
                    "user_question": user_question,
                    "chat_history": chat_history,
                    "knowledge_graph_context_str": knowledge_graph_context_str,
                },
            ) as event:
                condense_question_prompt = get_prompt_by_jinja2_template(
                    self.config.llm.condense_question_prompt,
                    question=user_question,
                    chat_history=chat_history,
                    graph_knowledges=knowledge_graph_context_str,
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
    ) -> StopEvent | RetrieveEvent:
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
            return StopEvent()

        return RetrieveEvent(
            question=clarity_result,
        )

    @step
    async def retrieve_relevant_chunks(
        self, ctx: Context, ev: RetrieveEvent
    ) -> GenerateAnswerEvent:
        refined_question = await ctx.get("refined_question")
        knowledge_base_ids = await ctx.get("knowledge_base_ids")

        with self.callback_manager.as_trace("retrieve_relevant_chunks"):
            with self.callback_manager.event(
                MyCBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: refined_question},
            ) as event:
                retriever = VectorSearchFusionRetriever(
                    db_session=self.db_session,
                    knowledge_base_ids=knowledge_base_ids,
                    llm=self.llm,
                    config=VectorSearchRetrieverConfig(
                        similarity_top_k=10,
                        oversampling_factor=5,
                        top_k=5,
                    ),
                )

                nodes_with_score = retriever.retrieve(QueryBundle(refined_question))

                event.on_end(
                    payload={
                        "nodes_with_score": nodes_with_score,
                    }
                )

        await ctx.set("nodes_with_score", nodes_with_score)

        return GenerateAnswerEvent()

    def _get_source_documents(
        self,
        db_session: Session,
        chunk_model: Type[SQLModel],
        nodes_with_score: List[NodeWithScore],
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
        source_chunks = db_session.exec(stmt).all()
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
        knowledge_graph_context_str = await ctx.get("knowledge_graph_context_str")

        with self.callback_manager.as_trace("generate_answer"):
            with self.callback_manager.event(
                MyCBEventType.SYNTHESIZE,
                payload={EventPayload.QUERY_STR: user_question},
            ) as event:
                text_qa_template = get_prompt_by_jinja2_template(
                    self.config.llm.text_qa_prompt,
                    current_date=datetime.now().strftime("%Y-%m-%d"),
                    graph_knowledges=knowledge_graph_context_str,
                    original_question=user_question,
                )
                synthesizer = get_response_synthesizer(
                    llm=self.llm,
                    text_qa_template=text_qa_template,
                    response_mode=ResponseMode.COMPACT,
                    callback_manager=self.callback_manager,
                    streaming=True,
                )
                response: StreamingResponse = synthesizer.synthesize(
                    user_question, nodes_with_score
                )

                response_text = ""
                for chunk in response.response_gen:
                    response_text += chunk
                    ctx.write_event_to_stream(GenerateAnswerStreamEvent(chunk=chunk))

                event.on_end(payload=response_text)

        return StopEvent(result=response_text)
