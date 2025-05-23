from sqlalchemy import inspection
from sqlmodel import Session

from app.models import KnowledgeBase
from app.models.chunk import get_kb_chunk_model
from app.models.entity import get_kb_entity_model
from app.rag.knowledge_base.config import get_kb_dspy_llm, get_kb_embed_model
from app.models.relationship import get_kb_relationship_model
from app.rag.indices.knowledge_graph.graph_store import TiDBGraphStore, TiDBGraphEditor
from app.rag.indices.vector_search.vector_store.tidb_vector_store import TiDBVectorStore


def get_kb_tidb_vector_store(session: Session, kb: KnowledgeBase) -> TiDBVectorStore:
    chunk_model = get_kb_chunk_model(kb)
    vector_store = TiDBVectorStore(chunk_model, session=session)
    return vector_store


def init_kb_tidb_vector_store(session: Session, kb: KnowledgeBase) -> TiDBVectorStore:
    vector_store = get_kb_tidb_vector_store(session, kb)
    vector_store.ensure_table_schema()
    return vector_store


def get_kb_tidb_graph_store(session: Session, kb: KnowledgeBase) -> TiDBGraphStore:
    dspy_lm = get_kb_dspy_llm(session, kb)
    embed_model = get_kb_embed_model(session, kb)
    entity_model = get_kb_entity_model(kb)
    relationship_model = get_kb_relationship_model(kb)
    inspection.inspect(relationship_model)
    chunk_model = get_kb_chunk_model(kb)

    graph_store = TiDBGraphStore(
        knowledge_base=kb,
        dspy_lm=dspy_lm,
        session=session,
        embed_model=embed_model,
        entity_db_model=entity_model,
        relationship_db_model=relationship_model,
        chunk_db_model=chunk_model,
    )
    return graph_store


def init_kb_tidb_graph_store(session: Session, kb: KnowledgeBase) -> TiDBGraphStore:
    graph_store = get_kb_tidb_graph_store(session, kb)
    graph_store.ensure_table_schema()
    return graph_store


def get_kb_tidb_graph_editor(session: Session, kb: KnowledgeBase) -> TiDBGraphEditor:
    entity_db_model = get_kb_entity_model(kb)
    relationship_db_model = get_kb_relationship_model(kb)
    embed_model = get_kb_embed_model(session, kb)
    return TiDBGraphEditor(
        knowledge_base_id=kb.id,
        entity_db_model=entity_db_model,
        relationship_db_model=relationship_db_model,
        embed_model=embed_model,
    )
