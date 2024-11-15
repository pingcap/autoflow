from sqlalchemy import inspection
from sqlmodel import Session

from app.models import KnowledgeBase
from app.models.chunk import get_kb_chunk_model
from app.models.entity import get_kb_entity_model
from app.rag.knowledge_base.config import get_kb_dspy_llm, get_kb_embed_model
from app.models.relationship import get_kb_relationship_model
from app.rag.knowledge_graph.graph_store import TiDBGraphStore
from app.rag.knowledge_graph.graph_store.tidb_graph_editor import TiDBGraphEditor
from app.rag.vector_store.tidb_vector_store import TiDBVectorStore


def get_kb_tidb_vector_store(session: Session, kb: KnowledgeBase) -> TiDBVectorStore:
    chunk_model = get_kb_chunk_model(kb)
    vector_store = TiDBVectorStore(
        session,
        chunk_db_model=chunk_model
    )
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

    graph_store = TiDBGraphStore(
        dspy_lm=dspy_lm,
        session=session,
        embed_model=embed_model,
        entity_db_model=entity_model,
        relationship_db_model=relationship_model,
    )
    return graph_store


def init_kb_tidb_graph_store(session: Session, kb: KnowledgeBase) -> TiDBGraphStore:
    graph_store = get_kb_tidb_graph_store(session, kb)
    graph_store.ensure_table_schema()
    return graph_store


def get_kb_tidb_graph_editor(session: Session, kb: KnowledgeBase) -> TiDBGraphEditor:
    entity_model = get_kb_entity_model(kb)
    relationship_model = get_kb_relationship_model(kb)
    init_kb_tidb_vector_store(session, kb)
    return TiDBGraphEditor(entity_model, relationship_model)
