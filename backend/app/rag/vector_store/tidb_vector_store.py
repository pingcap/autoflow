import logging
from typing import Any, List, Optional

import tidb_vector
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
import sqlalchemy
from sqlmodel import (
    SQLModel,
    Session,
    delete,
    select,
    asc,
)
from tidb_vector.sqlalchemy import VectorAdaptor

from app.core.db import engine
from app.models import Chunk

logger = logging.getLogger(__name__)


def node_to_relation_dict(node: BaseNode) -> dict:
    relations = {}
    for r_type, node_info in node.relationships.items():
        relations[r_type.name] = {
            "node_id": node_info.node_id,
            "node_type": node_info.node_type.name,
            "meta": node_info.metadata,
            "hash": node_info.hash,
        }
    return relations


class TiDBVectorStore(BasePydanticVectorStore):
    _session: Session = PrivateAttr()
    _owns_session: bool = PrivateAttr()
    _chunk_db_model: SQLModel = PrivateAttr()
    _table_name: str = PrivateAttr()
    _vector_dimension: int = PrivateAttr()

    stores_text: bool = True
    flat_metadata: bool = False

    def __init__(
        self,
        session: Optional[Session] = None,
        chunk_db_model: SQLModel = Chunk,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._session = session
        self._owns_session = session is None
        if self._session is None:
            self._session = Session(engine)

        self._chunk_db_model = chunk_db_model

    def ensure_table_schema(self) -> None:
        inspector = sqlalchemy.inspect(engine)
        table_name = self._chunk_db_model.__tablename__

        if table_name not in inspector.get_table_names():
            self._chunk_db_model.metadata.create_all(
                engine, tables=[self._chunk_db_model.__table__]
            )

            # Add HNSW index to accelerate ann queries.
            VectorAdaptor(engine).create_vector_index(
                self._chunk_db_model.embedding, tidb_vector.DistanceMetric.COSINE
            )

            logger.info(f"Chunk table <{table_name}> has been created successfully.")
        else:
            logger.info(
                f"Chunk table <{table_name}> is already exists, no action to do."
            )

    def drop_table_schema(self):
        inspector = sqlalchemy.inspect(engine)
        table_name = self._chunk_db_model.__tablename__

        if table_name in inspector.get_table_names():
            self._chunk_db_model.metadata.drop_all(
                self._session.connection(), tables=[self._chunk_db_model.__table__]
            )
            logger.info(f"Chunk table <{table_name}> has been dropped successfully.")
        else:
            logger.info(f"Chunk table <{table_name}> is not existed, not action to do.")

    def close_session(self) -> None:
        # Always call this method is necessary to make sure the session is closed
        if self._owns_session:
            self._session.close()

    @property
    def client(self) -> Any:
        """Get client."""
        return engine

    @classmethod
    def class_name(cls) -> str:
        return "TiDBVectorStore"

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes to the vector store.

        Args:
            nodes (List[BaseNode]): List of nodes to be added.
            **add_kwargs: Additional keyword arguments to be passed to the underlying storage.

        Returns:
            List[str]: List of node IDs that were added.
        """
        items = []
        for n in nodes:
            items.append(
                {
                    "id": n.node_id,
                    "hash": n.hash,
                    "text": n.get_content(metadata_mode=MetadataMode.NONE),
                    "meta": node_to_metadata_dict(n, remove_text=True),
                    "embedding": n.get_embedding(),
                    "document_id": n.ref_doc_id,
                    "relations": node_to_relation_dict(n),
                    "source_uri": add_kwargs.get("source_uri"),
                }
            )

        self._session.bulk_insert_mappings(self._chunk_db_model, items)
        self._session.commit()
        return [i["id"] for i in items]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete all nodes of a document from the vector store.

        Args:
            ref_doc_id (str): The reference document ID to be deleted.
            **delete_kwargs: Additional keyword arguments to be passed to the delete method.

        Returns:
            None
        """
        assert ref_doc_id.isdigit(), "ref_doc_id must be an integer."
        delete_stmt = delete(self._chunk_db_model).where(
            self._chunk_db_model.document_id == int(ref_doc_id)
        )
        self._session.exec(delete_stmt)
        self._session.commit()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Perform a similarity search with the given query embedding.

        Args:
            query (VectorStoreQuery): The query object containing the query data.
            **kwargs: Additional keyword arguments.

        Returns:
            VectorStoreQueryResult: The result of the similarity search.

        Raises:
            ValueError: If the query embedding is not provided.
        """
        # TODO:
        # - Support advanced query filters
        # - Support both pre-filter and post-filter
        if query.query_embedding is None:
            raise ValueError("Query embedding must be provided.")

        stmt = select(
            self._chunk_db_model.id,
            self._chunk_db_model.text,
            self._chunk_db_model.meta,
            self._chunk_db_model.embedding.cosine_distance(query.query_embedding).label(
                "distance"
            ),
        )

        if query.filters:
            for f in query.filters.filters:
                stmt = stmt.where(self._chunk_db_model.meta[f.key] == f.value)

        stmt = stmt.order_by(asc("distance")).limit(query.similarity_top_k)
        results = self._session.exec(stmt)

        nodes = []
        similarities = []
        ids = []
        for row in results:
            try:
                node = metadata_dict_to_node(row.meta)
                node.set_content(row.text)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                logger.warning(
                    "Failed to parse metadata dict, falling back to legacy logic."
                )
                node = TextNode(
                    id_=row.id,
                    text=row.text,
                    metadata=row.meta,
                )
            similarities.append((1 - row.distance) if row.distance is not None else 0)
            ids.append(row.id)
            nodes.append(node)
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )
