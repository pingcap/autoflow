import logging
from contextlib import contextmanager
from typing import Any, List, Optional

import sqlalchemy
import tidb_vector
from llama_index.core.vector_stores import MetadataFilters
from sqlalchemy import Engine
from sqlmodel import (
    desc,
    select,
    asc,
    Session,
)
from tidb_vector.sqlalchemy import VectorAdaptor

from autoflow.llms.embeddings import EmbeddingModel
from autoflow.models.document import Document
from autoflow.storage.doc_store.base import (
    DocumentStore,
    DocumentSearchResult,
    ChunkWithScore,
    D,
    C,
    DocumentSearchMethod,
)


logger = logging.getLogger(__name__)


class TiDBDocumentStore(DocumentStore[D, C]):
    def __init__(
        self,
        db_engine: Engine,
        embedding_model: EmbeddingModel,
        document_db_model: D,
        chunk_db_model: C,
    ) -> None:
        super().__init__()
        self._db_engine = db_engine
        self._embedding_model = embedding_model
        self._document_db_model = document_db_model
        self._chunk_db_model = chunk_db_model

    @classmethod
    def class_name(cls) -> str:
        return "TiDBDocumentStore"

    def ensure_table_schema(self) -> None:
        inspector = sqlalchemy.inspect(self._db_engine)
        existing_table_names = inspector.get_table_names()

        document_model = self._document_db_model
        document_table_name = document_model.__tablename__
        if document_table_name not in existing_table_names:
            document_model.metadata.create_all(
                self._db_engine, tables=[document_model.__table__]
            )
            logger.info(
                f"Document table <{document_table_name}> has been created successfully."
            )
        else:
            logger.info(
                f"Document table <{document_table_name}> is already exists, no action to do."
            )

        chunk_model = self._chunk_db_model
        chunk_table_name = chunk_model.__tablename__
        if chunk_table_name not in existing_table_names:
            chunk_model.metadata.create_all(
                self._db_engine, tables=[chunk_model.__table__]
            )
            VectorAdaptor(self._db_engine).create_vector_index(
                chunk_model.text_vec, tidb_vector.DistanceMetric.COSINE
            )
            logger.info(
                f"Chunk table <{chunk_table_name}> has been created successfully."
            )
        else:
            logger.info(
                f"Chunk table <{chunk_table_name}> is already exists, no action to do."
            )

    def drop_table_schema(self) -> None:
        inspector = sqlalchemy.inspect(self._db_engine)
        existed_table_names = inspector.get_table_names()

        document_model = self._document_db_model
        document_table_name = document_model.__tablename__
        if document_table_name in existed_table_names:
            document_model.metadata.drop_all(
                self._db_engine, tables=[document_model.__table__]
            )
            logger.info(
                f"Document table <{document_table_name}> has been dropped successfully."
            )
        else:
            logger.info(
                f"Document table <{document_table_name}> is not exists, no action to do."
            )

        chunk_model = self._chunk_db_model
        chunk_table_name = chunk_model.__tablename__
        if chunk_table_name in existed_table_names:
            chunk_model.metadata.drop_all(
                self._db_engine, tables=[chunk_model.__table__]
            )
            logger.info(
                f"Chunk table <{chunk_table_name}> has been dropped successfully."
            )
        else:
            logger.info(
                f"Chunk table <{chunk_table_name}> is not exists, no action to do."
            )

    @contextmanager
    def _session_scope(self, session: Optional[Session] = None, commit: bool = False):
        """Provide a transactional scope around a series of operations."""
        should_close = session is None
        session = session or Session(self._db_engine)

        try:
            yield session
            if commit:
                session.commit()
            else:
                session.flush()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if should_close:
                session.close()

    def add(self, documents: List[Document], **add_kwargs: Any) -> List[Document]:
        with Session(self._db_engine) as db_session:
            for doc in documents:
                db_session.merge(doc)
                db_session.add(doc)
            db_session.commit()
            for doc in documents:
                db_session.refresh(doc)
            return documents

    def delete(self, document_id: int) -> None:
        with Session(self._db_engine) as db_session:
            doc = db_session.get(self._document_db_model, document_id)
            if doc is None:
                raise ValueError("Document with id #{} not found".format(document_id))
            # TODO: Delete the chunks associated with the document.
            db_session.delete(doc)
            db_session.commit()

    def list(self) -> List[D]:
        with Session(self._db_engine) as db_session:
            query = select(self._document_db_model)
            return db_session.exec(query).all()

    def get(self, document_id: int) -> D:
        with Session(self._db_engine) as db_session:
            doc = db_session.get(self._document_db_model, document_id)
            if doc is None:
                raise ValueError("Document with id #{} not found".format(document_id))
            return doc

    def add_doc_chunks(self, chunks: List[C]) -> List[C]:
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self._embedding_model.get_text_embedding_batch(chunk_texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.text_vec = embedding

        with Session(self._db_engine) as db_session:
            db_session.bulk_save_objects(chunks)
            db_session.commit()
            return chunks

    # TODO: call the low-level database API.
    def search(
        self,
        query: str,
        search_method: List[DocumentSearchMethod] = [
            DocumentSearchMethod.VECTOR_SEARCH
        ],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        similarity_nprobe: Optional[int] = None,
        similarity_top_k: Optional[int] = 5,
        **kwargs: Any,
    ) -> DocumentSearchResult:
        # TODO: Support Hybrid search.
        with self._session_scope() as db_session:
            chunks_with_score = self._vector_search(
                query=query,
                # metadata_filters=query.metadata_filters,
                nprobe=similarity_nprobe,
                similarity_top_k=similarity_top_k,
                db_session=db_session,
            )
            # chunks_with_score = self._rerank_chunks(chunks_with_score)
            document_ids = {c.chunk.document_id for c in chunks_with_score}
            documents = self._list_documents_by_ids(
                list(document_ids), db_session=db_session
            )
            return DocumentSearchResult(
                chunks=chunks_with_score,
                documents=documents,
            )

    def _vector_search(
        self,
        query: str,
        # metadata_filters: Optional[MetadataFilters] = None,
        nprobe: Optional[int] = None,
        similarity_top_k: int = 5,
        db_session: Optional[Session] = None,
    ) -> List[ChunkWithScore]:
        query_embedding = self._embedding_model.get_query_embedding(query)
        nprobe = nprobe if nprobe else similarity_top_k * 10

        # Base query for vector similarity
        subquery = (
            select(
                self._chunk_db_model.id.label("chunk_id"),
                self._chunk_db_model.text_vec.cosine_distance(query_embedding).label(
                    "embedding_distance"
                ),
            )
            .order_by(asc("embedding_distance"))
            .limit(nprobe)
            .subquery()
        )

        # Main query with metadata filters
        query = select(
            self._chunk_db_model,
            (1 - subquery.c.embedding_distance).label("similarity_score"),
        ).join(subquery, self._chunk_db_model.id == subquery.c.chunk_id)

        # Apply metadata filters if provided
        # TODO: Implement metadata filters.

        # Apply final ordering and limit
        query = query.order_by(desc("similarity_score")).limit(similarity_top_k)

        with self._session_scope(db_session) as db_session:
            results = db_session.exec(query)
            return [
                ChunkWithScore(chunk=chunk, score=score) for chunk, score in results
            ]

    def _fulltext_search(
        self,
        query_str: str,
        metadata_filters: Optional[MetadataFilters] = None,
        top_k: Optional[int] = 5,
    ) -> List[ChunkWithScore]:
        raise NotImplementedError()

    def _rerank_chunks(
        self, chunks_with_score: List[ChunkWithScore]
    ) -> List[ChunkWithScore]:
        raise NotImplementedError("Reranking is not implemented for TiDBDocumentStore.")

    def _list_documents_by_ids(
        self, document_ids: List[int], db_session: Session = None
    ) -> List[D]:
        with self._session_scope(db_session) as db_session:
            query = select(self._document_db_model).where(
                self._document_db_model.id.in_(document_ids)
            )
            return db_session.exec(query).all()
