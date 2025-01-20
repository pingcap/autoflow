from datetime import datetime
import logging

from typing import List, Optional, Type
from fastapi.responses import StreamingResponse
from llama_index.core import get_response_synthesizer
from llama_index.llms.openai import OpenAI
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
from llama_index.core.embeddings import BaseEmbedding
from sqlmodel import Session, select, SQLModel

from app.models.chunk import get_kb_chunk_model
from app.rag.indices.knowledge_graph.retriever.base_retriever import (
    KnowledgeGraphRetriever,
)
from app.rag.indices.vector_search.schema import VectorSearchRetrieverConfig
from app.utils.jinja2 import get_prompt_by_jinja2_template
from app.rag.chat_config import ChatEngineConfig
from app.models import Document as DBDocument, KnowledgeBase
from app.rag.types import MyCBEventType
from app.rag.workflows.chat_app.events import (
    SearchKnowledgeGraphEvent,
    AggregateKGSearchResultEvent,
    RefineQuestionEvent,
    ClarifyQuestionEvent,
    RetrieveEvent,
    GenerateAnswerEvent,
    GenerateAnswerStreamEvent,
)
from app.site_settings import SiteSetting
from langfuse import Langfuse
from app.repositories import knowledge_base_repo
from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.indices.vector_search.base_retriever import VectorSearchRetriever
from app.utils import dspy

logger = logging.getLogger(__name__)


class ServiceContext:
    db_session: Session
    embed_model: BaseEmbedding
    llm: LLM
    fast_llm: OpenAI
    dspy_fast_lm: dspy.LM
    callback_manager: CallbackManager


class AppChatFlow(Workflow):
    """
    AppChatFlow is a standard chatting process for document-based document robots. It includes several key steps
    such as question rewriting, knowledge retrieval, and answer generation.
    """

    # Notice: ChatFlow should be reusable in different chat sessions, for example: the configuration of chat engine.
    # If you need to add session-specific variables, please use ctx.set() / ctx.get()
    config: ChatEngineConfig
    langfuse: Optional[Langfuse] = None

    def __init__(
        self,
        config: ChatEngineConfig,
        langfuse: Optional[Langfuse] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.langfuse = langfuse or Langfuse(
            host=SiteSetting.langfuse_host,
            secret_key=SiteSetting.langfuse_secret_key,
            public_key=SiteSetting.langfuse_public_key,
        )

    @step
    async def start_chat(
        self, ctx: Context, ev: StartEvent
    ) -> SearchKnowledgeGraphEvent | RefineQuestionEvent:
        for key, value in ev.items():
            await ctx.set(key, value)

        db_session = await ctx.get("db_session")

        llm = self.config.get_llama_llm(db_session)
        fast_llm = self.config.get_fast_llama_llm(db_session)
        fast_dspy_lm = self.config.get_fast_dspy_lm(db_session)

        # TODO: Support multiple knowledge base retrieval.
        knowledge_base = knowledge_base_repo.must_get(
            db_session, self.config.knowledge_base.linked_knowledge_base.id
        )
        await ctx.set("knowledge_base", knowledge_base)

        embed_model = get_kb_embed_model(db_session, knowledge_base)
        await ctx.set(
            "service_context",
            ServiceContext(
                db_session=db_session,
                embed_model=embed_model,
                llm=llm,
                fast_llm=fast_llm,
                dspy_fast_lm=fast_dspy_lm,
                callback_manager=CallbackManager([]),
            ),
        )

        if self.config.knowledge_graph.enabled:
            return SearchKnowledgeGraphEvent()
        else:
            return RefineQuestionEvent()

    @step
    async def search_knowledge_graph(
        self, ctx: Context, ev: SearchKnowledgeGraphEvent
    ) -> RefineQuestionEvent:
        user_question: str = await ctx.get("user_question")
        sc: ServiceContext = await ctx.get("service_context")

        with sc.callback_manager.as_trace("search_knowledge_graph"):
            with sc.callback_manager.event(
                MyCBEventType.GRAPH_SEMANTIC_SEARCH,
                payload={EventPayload.QUERY_STR: user_question},
            ) as event:
                kg_config = self.config.knowledge_graph
                knowledge_graph_retriever = KnowledgeGraphRetriever(
                    config=kg_config,
                    dspy_lm=sc.dspy_fast_lm,
                    callback_manager=sc.callback_manager,
                )
                entities, relationships = (
                    knowledge_graph_retriever.retrieve_knowledge_graph(
                        query_bundle=QueryBundle(
                            query_str=user_question,
                        )
                    )
                )
                event.on_end(
                    payload={"entities": entities, "relationships": relationships}
                )

        await ctx.set("knowledge_graph.entities", entities)
        await ctx.set("knowledge_graph.relationships", relationships)

        return RefineQuestionEvent(user_question=user_question)

    @step
    async def aggregate_knowledge_graph_search_result(
        self, ctx: Context, ev: AggregateKGSearchResultEvent
    ) -> RefineQuestionEvent:
        entities = ctx.get("knowledge_graph.entities")
        relationships = ctx.get("knowledge_graph.relationships")
        chunks = ctx.get("knowledge_graph.chunks")
        sc: ServiceContext = await ctx.get("service_context")

        with sc.callback_manager.as_trace("aggregate_knowledge_graph_search_result"):
            with sc.callback_manager.event(
                MyCBEventType.AGGREGATE_KNOWLEDGE_GRAPH_SEARCH_RESULT,
                payload={
                    "entities": entities,
                    "relationships": relationships,
                    "chunks": chunks,
                },
            ) as event:
                graph_data_source_ids = {
                    "entities": [e["id"] for e in entities],
                    "relationships": [r["id"] for r in relationships],
                }
                graph_knowledges = get_prompt_by_jinja2_template(
                    self.config.llm.normal_graph_knowledge,
                    entities=entities,
                    relationships=relationships,
                )
                graph_knowledges_context = graph_knowledges.template
                event.on_end(
                    payload={
                        "graph_data_source_ids": graph_data_source_ids,
                        "graph_knowledges_context": graph_knowledges_context,
                    }
                )

        await ctx.set("graph_data_source_ids", graph_data_source_ids)
        await ctx.set("graph_knowledges_context", graph_knowledges_context)

        return RefineQuestionEvent()

    @step
    async def refine_question(
        self, ctx: Context, ev: RefineQuestionEvent
    ) -> ClarifyQuestionEvent | RetrieveEvent:
        user_question = ctx.get("user_question")
        chat_history = ctx.get("chat_history", [])
        knowledge_graph_context = ctx.get("graph_knowledges_context")
        sc: ServiceContext = await ctx.get("service_context")

        with sc.callback_manager.as_trace("refine_question"):
            with sc.callback_manager.event(
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
                refined_question = sc.fast_llm.predict(condense_question_prompt)
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
        sc: ServiceContext = await ctx.get("service_context")

        with sc.callback_manager.as_trace("clarify_question"):
            with sc.callback_manager.event(
                MyCBEventType.CLARIFYING_QUESTION,
                payload={EventPayload.QUERY_STR: refined_question},
            ) as event:
                clarity_result = (
                    sc.fast_llm.structured_predict(
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
        kb: KnowledgeBase = await ctx.get("knowledge_base")
        sc: ServiceContext = await ctx.get("service_context")

        with sc.callback_manager.as_trace("retrieve_relevant_chunks"):
            with sc.callback_manager.event(
                MyCBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: refined_question},
            ) as event:
                chunk_model = get_kb_chunk_model(kb)
                retriever = VectorSearchRetriever(
                    knowledge_base_id=kb.id,
                    config=VectorSearchRetrieverConfig(
                        similarity_top_k=10,
                        oversampling_factor=5,
                        top_k=5,
                    ),
                )

                nodes_with_score = retriever.retrieve()
                source_documents = self._get_source_documents(
                    sc.db_session, chunk_model, nodes_with_score
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
        knowledge_graph_context = await ctx.get("knowledge_graph_context")
        sc: ServiceContext = await ctx.get("service_context")

        with sc.callback_manager.as_trace("generate_answer"):
            with sc.callback_manager.event(
                MyCBEventType.SYNTHESIZE,
                payload={EventPayload.QUERY_STR: user_question},
            ) as event:
                text_qa_template = get_prompt_by_jinja2_template(
                    self.config.llm.text_qa_prompt,
                    current_date=datetime.now().strftime("%Y-%m-%d"),
                    graph_knowledges=knowledge_graph_context,
                    original_question=user_question,
                )
                synthesizer = get_response_synthesizer(
                    llm=sc.llm,
                    text_qa_template=text_qa_template,
                    response_mode=ResponseMode.COMPACT,
                    callback_manager=sc.callback_manager,
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
