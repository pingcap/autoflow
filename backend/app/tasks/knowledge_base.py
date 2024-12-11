from celery.utils.log import get_task_logger
from sqlalchemy import delete
from sqlmodel import Session, select

from app.celery import app as celery_app
from app.core.db import engine
from app.exceptions import KBNotFound
from app.models import (
    Document, KnowledgeBaseDataSource, DataSource,
)
from app.models.knowledge_base import KnowledgeBase
from app.rag.datasource import get_data_source_loader
from app.repositories import knowledge_base_repo, document_repo
from .build_index import build_index_for_document
from ..models.chunk import get_kb_chunk_model
from ..models.entity import get_kb_entity_model
from ..models.relationship import get_kb_relationship_model
from ..rag.knowledge_base.index_store import get_kb_tidb_vector_store, get_kb_tidb_graph_store
from ..repositories.chunk import ChunkRepo
from ..repositories.graph import GraphRepo

logger = get_task_logger(__name__)

@celery_app.task
def import_documents_for_knowledge_base(kb_id: int):
    try:
        with Session(engine) as session:
            kb = knowledge_base_repo.must_get(session, kb_id)
            data_sources = kb.data_sources
            for data_source in data_sources:
                import_documents_from_kb_datasource(kb.id, data_source.id)

        logger.info(f"Successfully imported documents for knowledge base #{kb_id}")
    except KBNotFound:
        logger.error(f"Knowledge base #{kb_id} is not found")
    except Exception as e:
        logger.exception(f"Failed to import documents for knowledge base #{kb_id}", exc_info=e)


@celery_app.task
def import_documents_from_kb_datasource(kb_id: int, data_source_id: int):
    try:
        with Session(engine) as session:
            kb = knowledge_base_repo.must_get(session, kb_id)
            data_source = knowledge_base_repo.must_get_kb_datasource(session, kb, data_source_id)

            logger.info(f"Loading documents from data source #{data_source_id} for knowledge base #{kb_id}")
            loader = get_data_source_loader(
                session,
                kb_id,
                data_source.data_source_type,
                data_source.id,
                data_source.user_id,
                data_source.config,
            )

            for document in loader.load_documents():
                # When content of document is too long, truncate it when saving to database.
                # We can use original content to build index later.
                original_content = document.truncate_content(max_length=2*1024*1024)
                if original_content is not None:
                    logger.info(f"Truncate document #{document.id} from data source #{data_source_id}")
                    truncated_content_length = document.get_content_length()
                    logger.info(f"original content length: {len(original_content)}, truncated content length: {truncated_content_length}")

                session.add(document)
                session.commit()

                build_index_for_document.delay(kb_id, document.id, original_content)

        stats_for_knowledge_base.delay(kb_id)
        logger.info(f"Successfully imported documents for from datasource #{data_source_id}")
    except Exception as e:
        logger.exception(
            f"Failed to import documents from data source #{data_source_id} of knowledge base #{kb_id}",
            exc_info=e
        )


@celery_app.task
def stats_for_knowledge_base(kb_id: int):
    try:
        with Session(engine) as session:
            kb = knowledge_base_repo.must_get(session, kb_id)

            documents_total = knowledge_base_repo.count_documents(session, kb)
            data_sources_total = knowledge_base_repo.count_data_sources(session, kb)

            kb.documents_total = documents_total
            kb.data_sources_total = data_sources_total

            session.add(kb)
            session.commit()

        logger.info(f"Successfully running stats for knowledge base #{kb_id}")
    except KBNotFound:
        logger.error(f"Knowledge base #{kb_id} is not found")
    except Exception as e:
        logger.exception(f"Failed to run stats for knowledge base #{kb_id}", exc_info=e)


@celery_app.task
def purge_knowledge_base_related_resources(kb_id: int):
    """
    Purge all resources related to a knowledge base.

    Related resources:
        - documents
        - chunks
        - indexes
            - vector index
            - knowledge graph index
        - data sources
    """

    with Session(engine) as session:
        knowledge_base = knowledge_base_repo.must_get(session, kb_id, show_soft_deleted=True)
        assert knowledge_base.deleted_at is not None

        data_source_ids = [datasource.id for datasource in knowledge_base.data_sources]

        # Drop entities_{kb_id}, relationships_{kb_id} tables.
        tidb_graph_store = get_kb_tidb_graph_store(session, knowledge_base)
        tidb_graph_store.drop_table_schema()
        logger.info(f"Dropped tidb graph store of knowledge base #{kb_id} successfully.")

        # Drop chunks_{kb_id} table.
        tidb_vector_store = get_kb_tidb_vector_store(session, knowledge_base)
        tidb_vector_store.drop_table_schema()

        logger.info(f"Dropped tidb vector store of knowledge base #{kb_id} successfully.")

        # Delete documents.
        stmt = delete(Document).where(Document.knowledge_base_id == kb_id)
        session.exec(stmt)
        logger.info(f"Deleted documents of knowledge base #{kb_id} successfully.")

        # Delete data sources and links.
        if len(data_source_ids) > 0:
            stmt = delete(KnowledgeBaseDataSource).where(KnowledgeBaseDataSource.knowledge_base_id == kb_id)
            session.exec(stmt)
            logger.info(f"Deleted linked data sources of knowledge base #{kb_id} successfully.")

            stmt = delete(DataSource).where(DataSource.id.in_(data_source_ids))
            session.exec(stmt)
            logger.info(f"Deleted data sources {', '.join([f'#{did}' for did in data_source_ids])} successfully.")

        # Delete knowledge base.
        session.delete(knowledge_base)
        logger.info(f"Deleted knowledge base #{kb_id} successfully.")

        session.commit()


@celery_app.task
def purge_kb_datasource_related_resources(kb_id: int, datasource_id: int):
    """
    Purge all resources related to the deleted datasource in the knowledge base.
    """

    with Session(engine) as session:
        kb = knowledge_base_repo.must_get(session, kb_id, show_soft_deleted=True)
        datasource = knowledge_base_repo.must_get_kb_datasource(session, kb, datasource_id, show_soft_deleted=True)
        assert datasource.deleted_at is not None

        chunk_model = get_kb_chunk_model(kb)
        entity_model = get_kb_entity_model(kb)
        relationship_model = get_kb_relationship_model(kb)

        chunk_repo = ChunkRepo(chunk_model)
        graph_repo = GraphRepo(entity_model, relationship_model, chunk_model)

        graph_repo.delete_data_source_relationships(session, datasource_id)
        logger.info(f"Deleted relationships generated by chunks from data source #{datasource_id} successfully.")

        graph_repo.delete_orphaned_entities(session)
        logger.info(f"Deleted orphaned entities successfully.")

        chunk_repo.delete_by_datasource(session, datasource_id)
        logger.info(f"Deleted chunks from data source #{datasource_id} successfully.")

        document_repo.delete_by_datasource(session, datasource_id)
        logger.info(f"Deleted documents from data source #{datasource_id} successfully.")

        session.delete(datasource)
        logger.info(f"Deleted data source #{datasource_id} successfully.")

        session.commit()

    stats_for_knowledge_base.delay(kb_id)
