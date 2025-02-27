import logging
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

from autoflow.models.embeddings import EmbeddingModel
from autoflow.db_models.document import Document
from autoflow.stores.document_store.base import (
    DocumentStore,
    DocumentSearchQuery,
    DocumentSearchResult,
    ChunkWithScore,
    D,
    C,
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
                chunk_model.embedding, tidb_vector.DistanceMetric.COSINE
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

    def add(self, documents: List[Document], **add_kwargs: Any) -> List[Document]:
        with Session(self._db_engine) as db_session:
            db_session.bulk_save_objects(documents)
            db_session.commit()
            db_session.refresh(documents)
            return documents

    def delete(self, document_id: int) -> None:
        with Session(self._db_engine) as db_session:
            doc = db_session.get(self._document_db_model, document_id)
            if doc is None:
                raise ValueError("Document with id #{} not found".format(document_id))
            # TODO: Delete the chunks associated with the document.
            db_session.delete(doc)
            db_session.commit()

    def list(self) -> List[Document]:
        with Session(self._db_engine) as db_session:
            query = select(self._document_db_model)
            return db_session.exec(query).all()

    def get(self, document_id: int) -> D:
        with Session(self._db_engine) as db_session:
            doc = db_session.get(self._document_db_model, document_id)
            if doc is None:
                raise ValueError("Document with id #{} not found".format(document_id))
            return doc

    # TODO: call the low-level database API.
    def search(self, query: DocumentSearchQuery, **kwargs: Any) -> DocumentSearchResult:
        if query.query_embedding is None:
            query.query_embedding = self._embedding_model.get_query_embedding(
                query.query_str
            )

        chunks_with_score = self._vector_search(
            query_embedding=query.query_embedding,
            metadata_filters=query.metadata_filters,
            nprobe=query.nprobe,
            similarity_top_k=query.similarity_top_k,
        )
        chunks_with_score = self._rerank_chunks(chunks_with_score)
        documents = {c.chunk.document for c in chunks_with_score}
        return DocumentSearchResult(
            chunks=chunks_with_score,
            documents=list(documents),
        )

    def _vector_search(
        self,
        query_embedding: List[float],
        metadata_filters: Optional[MetadataFilters] = None,
        nprobe: Optional[int] = None,
        similarity_top_k: Optional[int] = 5,
    ) -> List[ChunkWithScore]:
        nprobe = nprobe if nprobe else similarity_top_k * 10

        # Base query for vector similarity
        subquery = (
            select(
                self._chunk_db_model.id.label("chunk_id"),
                self._chunk_db_model.embedding.cosine_distance(query_embedding).label(
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

        with Session(self._db_engine) as db_session:
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
