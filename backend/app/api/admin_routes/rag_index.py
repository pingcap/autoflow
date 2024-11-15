from fastapi import APIRouter
from sqlmodel import select, func

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.models import Document, Chunk, Relationship, Entity

router = APIRouter()


@router.get("/admin/rag/index-progress", deprecated=True)
def status(session: SessionDep, user: CurrentSuperuserDep):
    statement = (
        select(Chunk.index_status, func.count(Chunk.id))
        .group_by(Chunk.index_status)
        .order_by(Chunk.index_status)
    )
    status = session.exec(statement).all()
    chunk_index_status = {s: c for s, c in status}

    statement = (
        select(Document.index_status, func.count(Document.id))
        .group_by(Document.index_status)
        .order_by(Document.index_status)
    )
    status = session.exec(statement).all()
    document_index_status = {s: c for s, c in status}

    documents_count = session.scalar(select(func.count(Document.id)))
    chunks_count = session.scalar(select(func.count(Chunk.id)))
    entities_count = session.scalar(select(func.count(Entity.id)))
    relationships_count = session.scalar(select(func.count(Relationship.id)))

    return {
        "kg_index": chunk_index_status,
        "vector_index": document_index_status,
        "documents": {
            "total": documents_count,
        },
        "chunks": {
            "total": chunks_count,
        },
        "entities": {
            "total": entities_count,
        },
        "relationships": {
            "total": relationships_count,
        },
    }
