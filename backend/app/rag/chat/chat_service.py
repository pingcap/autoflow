from http import HTTPStatus
import logging

from typing import Generator, List, Optional
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import text, delete
from sqlmodel import Session, select, func

from app.api.routes.models import (
    RequiredConfigStatus,
    OptionalConfigStatus,
    NeedMigrationStatus,
)
from app.models import (
    User,
    ChatVisibility,
    Chat as DBChat,
    ChatMessage as DBChatMessage,
    KnowledgeBase as DBKnowledgeBase,
    RerankerModel as DBRerankerModel,
    ChatEngine,
)
from app.models.recommend_question import RecommendQuestion
from app.rag.chat.retrieve.retrieve_flow import RetrieveFlow, SourceDocument
from app.rag.chat.stream_protocol import ChatEvent
from app.rag.retrievers.knowledge_graph.schema import (
    KnowledgeGraphRetrievalResult,
    StoredKnowledgeGraph,
    RetrievedSubGraph,
)
from app.rag.knowledge_base.index_store import get_kb_tidb_graph_store
from app.repositories import knowledge_base_repo

from app.rag.chat.config import (
    ChatEngineConfig,
)
from app.rag.types import (
    ChatEventType,
    ChatMessageSate,
)
from app.repositories import chat_engine_repo
from app.repositories.embedding_model import embed_model_repo
from app.repositories.llm import llm_repo
from app.site_settings import SiteSetting
from app.utils.jinja2 import get_prompt_by_jinja2_template

logger = logging.getLogger(__name__)


class ChatResult(BaseModel):
    chat_id: UUID
    message_id: int
    content: str
    trace: Optional[str] = None
    sources: Optional[List[SourceDocument]] = []


def get_final_chat_result(
    generator: Generator[ChatEvent | str, None, None],
) -> ChatResult:
    trace, sources, content = None, [], ""
    chat_id, message_id = None, None
    for m in generator:
        if not isinstance(m, ChatEvent):
            continue
        if m.event_type == ChatEventType.MESSAGE_ANNOTATIONS_PART:
            if m.payload.state == ChatMessageSate.SOURCE_NODES:
                sources = m.payload.context
        elif m.event_type == ChatEventType.TEXT_PART:
            content += m.payload
        elif m.event_type == ChatEventType.DATA_PART:
            chat_id = m.payload.chat.id
            message_id = m.payload.assistant_message.id
            trace = m.payload.assistant_message.trace_url
        elif m.event_type == ChatEventType.ERROR_PART:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=m.payload,
            )
        else:
            pass
    return ChatResult(
        chat_id=chat_id,
        message_id=message_id,
        trace=trace,
        sources=sources,
        content=content,
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
    db_session: Session,
    chat_message: DBChatMessage,
    engine_config: ChatEngineConfig,
) -> Optional[KnowledgeGraphRetrievalResult]:
    if not chat_message.graph_data:
        return None

    graph_data = chat_message.graph_data

    # For forward compatibility.
    if "version" not in graph_data:
        kb = engine_config.get_knowledge_bases(db_session)[0]
        graph_store = get_kb_tidb_graph_store(db_session, kb)
        return graph_store.get_subgraph_by_relationship_ids(graph_data["relationships"])

    # Stored Knowledge Graph -> Retrieved Knowledge Graph
    stored_kg = StoredKnowledgeGraph.model_validate(graph_data)
    if stored_kg.knowledge_base_id is not None:
        kb = knowledge_base_repo.must_get(db_session, stored_kg.knowledge_base_id)
        graph_store = get_kb_tidb_graph_store(db_session, kb)
        retrieved_kg = graph_store.get_subgraph_by_relationship_ids(
            ids=stored_kg.relationships, query=stored_kg.query
        )
        return retrieved_kg
    elif stored_kg.knowledge_base_ids is not None:
        kg_store_map = {}
        knowledge_base_set = set()
        relationship_set = set()
        entity_set = set()
        subgraphs = []

        for kb_id in stored_kg.knowledge_base_ids:
            kb = knowledge_base_repo.must_get(db_session, kb_id)
            knowledge_base_set.add(kb.to_descriptor())
            kg_store = get_kb_tidb_graph_store(db_session, kb)
            kg_store_map[kb_id] = kg_store

        for stored_subgraph in stored_kg.subgraphs:
            kg_store = kg_store_map.get(stored_subgraph.knowledge_base_id)
            if kg_store is None:
                continue
            relationship_ids = stored_subgraph.relationships
            subgraph = kg_store.get_subgraph_by_relationship_ids(
                ids=relationship_ids,
                query=stored_kg.query,
            )
            relationship_set.update(subgraph.relationships)
            entity_set.update(subgraph.entities)
            subgraphs.append(
                RetrievedSubGraph(
                    **subgraph.model_dump(),
                )
            )

        return KnowledgeGraphRetrievalResult(
            query=stored_kg.query,
            knowledge_bases=list(knowledge_base_set),
            relationships=list(relationship_set),
            entities=list(entity_set),
            subgraphs=subgraphs,
        )
    else:
        return None


def get_chat_message_subgraph(
    db_session: Session, chat_message: DBChatMessage
) -> KnowledgeGraphRetrievalResult:
    chat_engine: ChatEngine = chat_message.chat.engine
    engine_name = chat_engine.name
    engine_config = ChatEngineConfig.load_from_db(db_session, chat_engine.name)

    # Try to get subgraph from `chat_message.graph_data`.
    try:
        knowledge_graph = get_graph_data_from_chat_message(
            db_session, chat_message, engine_config
        )
        if knowledge_graph is not None:
            return knowledge_graph
    except Exception as e:
        logger.error(
            f"Failed to get subgraph from chat_message.graph_data: {e}", exc_info=True
        )

    # Try to get subgraph based on the chat message content.
    # Notice: it use current chat engine config, not the snapshot stored in chat_message.
    retriever = RetrieveFlow(
        db_session=db_session,
        engine_name=engine_name,
        engine_config=engine_config,
    )
    knowledge_graph, _ = retriever.search_knowledge_graph(chat_message.content)
    return knowledge_graph


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
