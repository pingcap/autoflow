from sqlmodel import Session

from app.models import KnowledgeBase
from app.models.chunk import get_kb_chunk_model
from app.rag.indices.knowledge_graph.graph_store import TiDBGraphStore, TiDBGraphEditor
from app.rag.indices.vector_search.vector_store.tidb_vector_store import TiDBVectorStore


def get_kb_tidb_vector_store(session: Session, kb: KnowledgeBase) -> TiDBVectorStore:
    chunk_model = get_kb_chunk_model(kb)
    vector_store = TiDBVectorStore(session, chunk_db_model=chunk_model)
    return vector_store


def init_kb_tidb_vector_store(session: Session, kb: KnowledgeBase) -> TiDBVectorStore:
    vector_store = get_kb_tidb_vector_store(session, kb)
    vector_store.ensure_table_schema()
    return vector_store


def get_kb_tidb_graph_store(session: Session, kb: KnowledgeBase) -> TiDBGraphStore:
    return TiDBGraphStore.from_knowledge_base(kb, session)


def init_kb_tidb_graph_store(session: Session, kb: KnowledgeBase) -> TiDBGraphStore:
    graph_store = get_kb_tidb_graph_store(session, kb)
    graph_store.ensure_table_schema()
    return graph_store


def get_kb_graph_editor(session: Session, kb: KnowledgeBase) -> TiDBGraphEditor:
    graph_store = get_kb_tidb_graph_store(session, kb)
    return TiDBGraphEditor(session, graph_store)
