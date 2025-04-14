import logging
from uuid import UUID
from typing import Any, Dict, List, Optional, Type

from pydantic import PrivateAttr

from pytidb import TiDBClient, Table
from pytidb.embeddings import EmbeddingFunction
from pytidb.schema import TableModel, Field, Column, Relationship as SQLRelationship
from pytidb.datatype import Vector, JSON
from pytidb.search import SearchType
from sqlalchemy.dialects.mysql import LONGTEXT

from autoflow.data_types import DataType
from autoflow.models.embedding_models import EmbeddingModel
from autoflow.orms.base import UUIDBaseModel
from autoflow.storage.doc_store.types import (
    Document,
    DocumentDescriptor,
    Chunk,
    RetrievedChunk,
    DocumentSearchResult,
)
from autoflow.types import SearchMode
from autoflow.storage.doc_store.base import DocumentStore


logger = logging.getLogger(__name__)


def dynamic_create_models(
    namespace: Optional[str] = None,
    embedding_model: Optional[EmbeddingModel] = None,
    vector_dims: Optional[int] = None,
) -> tuple[Type[TableModel], Type[TableModel]]:
    if embedding_model is None and vector_dims is None:
        raise ValueError("Either `embedding_model` or `vector_dims` must be specified")

    # Determine the table names.
    suffix = f"_{namespace}" if namespace else ""
    document_table_name = f"documents{suffix}"
    document_model_name = f"DBDocument{suffix}"
    chunk_table_name = f"chunks{suffix}"
    chunk_model_name = f"DBChunk{suffix}"

    # Initialize the document table model.
    class DBDocument(UUIDBaseModel):
        hash: str = Field(max_length=128)
        name: str = Field(max_length=256)
        content: str = Field(sa_column=Column(LONGTEXT))
        data_type: Optional[DataType] = Field(default=None)
        meta: dict = Field(default_factory=dict, sa_column=Column(JSON))

    document_model = type(
        document_model_name,
        (DBDocument,),
        {
            "__tablename__": document_table_name,
            "__table_args__": {"extend_existing": True},
        },
        table=True,
    )

    # Initialize the chunk table model.
    if embedding_model is not None:
        embed_fn = EmbeddingFunction(
            model_name=embedding_model.model_name,
            dimensions=embedding_model.dimensions,
            api_key=embedding_model.api_key,
            api_base=embedding_model.api_base,
            timeout=embedding_model.timeout,
        )
        vector_field = embed_fn.VectorField(source_field="text")
    else:
        vector_field = Field(default=None, sa_column=Column(Vector(vector_dims)))

    class DBChunk(UUIDBaseModel):
        text: str = Field(sa_column=Column(LONGTEXT))
        text_vec: Optional[Any] = vector_field
        document_id: UUID = Field(foreign_key=f"{document_table_name}.id")

    chunk_model = type(
        chunk_model_name,
        (DBChunk,),
        {
            "__tablename__": chunk_table_name,
            "__table_args__": {"extend_existing": True},
            "__annotations__": {
                "document": Optional[document_model],
            },
            "document": SQLRelationship(
                sa_relationship_kwargs={
                    "cascade": "all, delete",
                },
            ),
        },
        table=True,
    )

    return document_model, chunk_model


class TiDBDocumentStore(DocumentStore):
    _client: TiDBClient = PrivateAttr()
    _document_db_model: Type[Type[TableModel]] = PrivateAttr()
    _document_table: Table = PrivateAttr()
    _chunk_db_model: Type[Type[TableModel]] = PrivateAttr()
    _chunk_table: Table = PrivateAttr()

    def __init__(
        self,
        client: TiDBClient,
        namespace: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_dims: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._db_engine = self._client.db_engine
        self._embedding_model = embedding_model
        self._init_store(namespace, vector_dims)

    @classmethod
    def class_name(cls) -> str:
        return "TiDBDocumentStore"

    def _init_store(
        self, namespace: Optional[str] = None, vector_dims: Optional[int] = None
    ):
        self._document_db_model, self._chunk_db_model = dynamic_create_models(
            namespace=namespace,
            vector_dims=vector_dims,
            embedding_model=self._embedding_model,
        )
        self._document_table = self._client.create_table(schema=self._document_db_model)
        self._chunk_table = self._client.create_table(schema=self._chunk_db_model)

    # Document Operations.

    def add(self, documents: List[Document]) -> List[Document]:
        """
        Add documents.
        """
        return_documents = []
        for doc in documents:
            db_document = self._document_db_model(**doc.model_dump(exclude={"chunks"}))
            db_document = self._document_table.insert(db_document)

            return_chunks = []
            if doc.chunks is not None and len(doc.chunks) > 0:
                db_chunks = self.add_doc_chunks(db_document.id, doc.chunks)
                return_chunks = [
                    Chunk(**db_chunk.model_dump(exclude={"document"}))
                    for db_chunk in db_chunks
                ]

            return_documents.append(
                Document(**db_document.model_dump(), chunks=return_chunks)
            )

        return return_documents

    def update(self, document_id: UUID, update: Dict[str, Any]) -> None:
        """
        Update documents.
        """
        self._document_table.update(update, {"id": document_id})

    def delete(self, document_id: UUID) -> None:
        """
        Delete document by id.

        Note: The related chunks will also be deleted by cascade.

        Args:
            document_id: The id of the document to delete.
        """
        return self._document_table.delete({"id": document_id})

    def get(self, document_id: UUID) -> Document:
        """
        Get document by id.
        """
        db_document = self._document_table.get(document_id)
        return Document(**db_document.model_dump())

    # TODO: Support pagination.
    def list(self, filters: Dict[str, Any] = None) -> List[Document]:
        """
        List all documents.
        """
        db_documents = self._document_table.query(filters)
        return [Document(**d.model_dump()) for d in db_documents]

    def search(
        self,
        query: str | List[float],
        mode: SearchMode = "vector",
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        num_candidate: Optional[int] = None,
        full_document: Optional[bool] = None,
    ) -> DocumentSearchResult:
        # TODO: Support Fulltext search.
        # TODO: Support Hybrid search.
        if mode != "vector":
            raise NotImplementedError(
                ".search() only supports vector search currently, fulltext and hybird search will be coming soon."
            )

        db_chunks = (
            self._chunk_table.search(query, query_type=SearchType.VECTOR_SEARCH)
            .distance_threshold(
                (1 - similarity_threshold) if similarity_threshold is not None else None
            )
            .num_candidate(num_candidate)
            .limit(top_k)
            .to_pydantic(with_score=True)
        )
        document_ids = [c.document_id for c in db_chunks]
        db_documents = self.list(
            {
                "id": {"$in": document_ids},
            }
        )
        return self._convert_to_retrieval_result(db_chunks, db_documents, full_document)

    def _convert_to_retrieval_result(
        self,
        db_chunks: List[TableModel],
        db_documents: List[TableModel],
        full_document: bool,
    ) -> DocumentSearchResult:
        return DocumentSearchResult(
            chunks=[
                RetrievedChunk(
                    **c.hit.model_dump(),
                    similarity_score=c.similarity_score,
                    score=c.score,
                )
                for c in db_chunks
            ],
            documents=[
                Document(**d.model_dump())
                if full_document
                else DocumentDescriptor(**d.model_dump())
                for d in db_documents
            ],
        )

    # Chunk Operations.

    def add_doc_chunks(self, document_id: UUID, chunks: List[Chunk]) -> List[Chunk]:
        """
        Add document chunks.
        """
        db_chunks = [
            self._chunk_db_model(
                **c.model_dump(exclude={"document_id"}), document_id=document_id
            )
            for c in chunks
        ]
        db_chunks = self._chunk_table.bulk_insert(db_chunks)
        return [Chunk(**c.model_dump(exclude={"document"})) for c in db_chunks]

    def list_doc_chunks(self, document_id: UUID) -> List[Chunk]:
        """
        List document chunks.
        """
        return self._chunk_table.query({"document_id": document_id})

    def get_chunk(self, chunk_id: UUID) -> Chunk:
        """
        Get chunk by id.
        """
        chunk = self._chunk_table.get(chunk_id)
        return Chunk(**chunk.model_dump(exclude={"document"}))

    def delete_chunk(self, chunk_id: UUID) -> None:
        """
        Delete document chunk.
        """
        return self._chunk_table.delete({"id": chunk_id})

    def update_chunk(self, chunk_id: UUID, update: Dict[str, Any]) -> Chunk:
        """
        Update chunk.
        """
        self._chunk_table.update(update, {"id": chunk_id})
        return self.get_chunk(chunk_id)

    # Document Store Operations.

    def recreate(self) -> None:
        self._client.drop_table(self._chunk_table.table_name)
        self._client.drop_table(self._document_table.table_name)
        self._document_table = self._client.create_table(schema=self._document_db_model)
        self._chunk_table = self._client.create_table(schema=self._chunk_db_model)

    def reset(self) -> None:
        with self._client.session():
            self._client.execute("SET FOREIGN_KEY_CHECKS = 0")
            self._chunk_table.truncate()
            self._document_table.truncate()
            self._client.execute("SET FOREIGN_KEY_CHECKS = 1")
