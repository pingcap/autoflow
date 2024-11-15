from celery.utils.log import get_task_logger
from sqlalchemy import delete
from sqlmodel import Session

from app.celery import app as celery_app
from app.core.db import engine
from app.exceptions import KnowledgeBaseNotFoundError
from app.models import (
    Document, KnowledgeBaseDataSource, DataSource,
)
from app.models.knowledge_base import KnowledgeBase
from app.rag.datasource import get_data_source_loader
from app.repositories import knowledge_base_repo
from .build_index import build_index_for_document
from ..rag.knowledge_base.index_store import get_kb_tidb_vector_store, get_kb_tidb_graph_store

logger = get_task_logger(__name__)

@celery_app.task
def import_documents_for_knowledge_base(knowledge_base_id: int):
    try:
        with Session(engine) as session:
            kb = knowledge_base_repo.must_get(session, knowledge_base_id)
            data_sources = kb.data_sources
            for data_source in data_sources:
                loader = get_data_source_loader(
                    session,
                    knowledge_base_id,
                    data_source.data_source_type,
                    data_source.id,
                    data_source.user_id,
                    data_source.config,
                )
                for document in loader.load_documents():
                    session.add(document)
                    session.commit()

                    build_index_for_document.delay(kb.id, document.id)
        logger.info(f"Successfully imported documents for knowledge base #{knowledge_base_id}")
    except KnowledgeBaseNotFoundError:
        logger.error(f"Knowledge base #{knowledge_base_id} is not found")
    except Exception as e:
        logger.exception(f"Failed to import documents for knowledge base #{knowledge_base_id}", exc_info=e)


@celery_app.task
def purge_knowledge_base_related_resources(knowledge_base_id: int):
    """Purge all resources related to a knowledge base.

    Related resources:
        - documents
        - chunks
        - indexes
            - vector index
            - knowledge graph index
        - data sources
    """

    with Session(engine) as session:
        knowledge_base = knowledge_base_repo.get(session, knowledge_base_id, include_soft_deleted=True)
        if knowledge_base is None:
            logger.error(f"Knowledge base with id {knowledge_base_id} not found")
            return

        assert knowledge_base.deleted_at is not None

        data_source_ids = [datasource.id for datasource in knowledge_base.data_sources]

        # Drop entities_{kb_id}, relationships_{kb_id} tables.
        tidb_graph_store = get_kb_tidb_graph_store(session, knowledge_base)
        tidb_graph_store.drop_table_schema()
        logger.info(f"Dropped tidb graph store of knowledge base #{knowledge_base_id} successfully.")

        # Drop chunks_{kb_id} table.
        tidb_vector_store = get_kb_tidb_vector_store(session, knowledge_base)
        tidb_vector_store.drop_table_schema()

        logger.info(f"Dropped tidb vector store of knowledge base #{knowledge_base_id} successfully.")

        # Delete documents.
        stmt = delete(Document).where(Document.knowledge_base_id == knowledge_base_id)
        session.exec(stmt)
        logger.info(f"Deleted documents of knowledge base #{knowledge_base_id} successfully.")

        # Delete data source links.
        stmt = delete(KnowledgeBaseDataSource).where(KnowledgeBase.id == knowledge_base_id)
        session.exec(stmt)
        logger.info(f"Deleted linked data sources of knowledge base #{knowledge_base_id} successfully.")

        # Delete data sources.
        stmt = delete(DataSource).where(DataSource.id.in_(data_source_ids))
        session.exec(stmt)
        logger.info(f"Deleted data sources {', '.join([f'#{did}' for did in data_source_ids])} successfully.")

        # Delete knowledge base.
        stmt = delete(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id)
        session.exec(stmt)
        logger.info(f"Deleted knowledge base #{knowledge_base_id} successfully.")

        session.commit()
