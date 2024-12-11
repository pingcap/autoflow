import traceback
from uuid import UUID
from sqlmodel import Session
from celery.utils.log import get_task_logger

from app.celery import app as celery_app
from app.core.db import engine
from app.models import (
    Document as DBDocument,
    DocIndexTaskStatus,
    KgIndexStatus,
)
from app.models.chunk import get_kb_chunk_model
from app.models.knowledge_base import IndexMethod
from app.rag.build_index import IndexService
from app.rag.knowledge_base.config import get_kb_llm, get_kb_embed_model
from app.repositories import knowledge_base_repo
from app.repositories.chunk import ChunkRepo

logger = get_task_logger(__name__)


# TODO: refactor: divide into two tasks: build_vector_index_for_document and build_kg_index_for_document


@celery_app.task(bind=True)
def build_index_for_document(self, knowledge_base_id: int, document_id: int):
    # Pre-check before building index.
    with Session(engine, expire_on_commit=False) as session:
        kb = knowledge_base_repo.must_get(session, knowledge_base_id)

        # Check document.
        db_document = session.get(DBDocument, document_id)
        if db_document is None:
            logger.error(f"Document #{document_id} is not found")
            return

        if db_document.index_status not in (
            DocIndexTaskStatus.PENDING,
            DocIndexTaskStatus.NOT_STARTED,
        ):
            logger.info(f"Document #{document_id} is not in pending state")
            return

        # Init knowledge base index service。
        try:
            llm = get_kb_llm(session, kb)
            embed_model = get_kb_embed_model(session, kb)
            index_service = IndexService(llm, embed_model, kb)
        except ValueError as e:
            # LLM may not be available yet(eg. bootstrapping), retry after specified time
            logger.warning(
                f"Failed to init index service for document #{document_id} (retry task after 1 minute): {e}"
            )
            raise self.retry(countdown=60)

        db_document.index_status = DocIndexTaskStatus.RUNNING
        session.add(db_document)
        session.commit()

    # Build vector index.
    try:
        with Session(engine) as index_session:
            index_service.build_vector_index_for_document(index_session, db_document)

        with Session(engine) as session:
            db_document.index_status = DocIndexTaskStatus.COMPLETED
            session.add(db_document)
            session.commit()
            logger.info(f"Built vector index for document #{document_id} successfully.")
    except Exception:
        with Session(engine) as session:
            error_msg = traceback.format_exc()
            logger.error(
                f"Failed to build vector index for document {document_id}: {error_msg}"
            )
            db_document.index_status = DocIndexTaskStatus.FAILED
            db_document.index_result = error_msg
            session.add(db_document)
            session.commit()
        return

    # Build knowledge graph index.
    with Session(engine, expire_on_commit=False) as session:
        kb = knowledge_base_repo.must_get(session, knowledge_base_id)
        if IndexMethod.KNOWLEDGE_GRAPH not in kb.index_methods:
            return

        chunk_repo = ChunkRepo(get_kb_chunk_model(kb))
        chunks = chunk_repo.get_document_chunks(session, document_id)
        for chunk in chunks:
            build_kg_index_for_chunk.delay(knowledge_base_id, chunk.id)


@celery_app.task
def build_kg_index_for_chunk(knowledge_base_id: int, chunk_id: UUID):
    with Session(engine, expire_on_commit=False) as session:
        kb = knowledge_base_repo.must_get(session, knowledge_base_id)

        # Check chunk.
        chunk_model = get_kb_chunk_model(kb)
        db_chunk = session.get(chunk_model, chunk_id)
        if db_chunk is None:
            logger.error(f"Chunk #{chunk_id} is not found")
            return

        if db_chunk.index_status not in (
            KgIndexStatus.PENDING,
            KgIndexStatus.NOT_STARTED,
        ):
            logger.info(f"Chunk #{chunk_id} is not in pending state")
            return

        # Init knowledge base index service。
        llm = get_kb_llm(session, kb)
        embed_model = get_kb_embed_model(session, kb)
        index_service = IndexService(llm, embed_model, kb)

        db_chunk.index_status = KgIndexStatus.RUNNING
        session.add(db_chunk)
        session.commit()

    try:
        with Session(engine) as index_session:
            index_service.build_kg_index_for_chunk(index_session, db_chunk)

        with Session(engine) as session:
            db_chunk.index_status = KgIndexStatus.COMPLETED
            session.add(db_chunk)
            session.commit()
            logger.info(
                f"Built knowledge graph index for chunk #{chunk_id} successfully."
            )
    except Exception:
        with Session(engine) as session:
            error_msg = traceback.format_exc()
            logger.error(
                f"Failed to build knowledge graph index for chunk #{chunk_id}: {error_msg}"
            )
            db_chunk.index_status = KgIndexStatus.FAILED
            db_chunk.index_result = error_msg
            session.add(db_chunk)
            session.commit()
