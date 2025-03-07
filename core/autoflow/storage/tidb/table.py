import logging
from typing import Optional, List, Any, Dict

import sqlalchemy
from sqlalchemy import Engine, update, text
from sqlalchemy.orm import Session, DeclarativeMeta
from sqlmodel.main import SQLModelMetaclass
from tidb_vector.sqlalchemy import VectorAdaptor

from autoflow.storage.tidb.base import Base
from autoflow.storage.tidb.constants import VectorDataType, TableModel, DistanceMetric
from autoflow.storage.tidb.embeddings import EmbeddingModel
from autoflow.storage.tidb.query import QueryType, TiDBVectorQuery
from autoflow.storage.tidb.utils import (
    build_filter_clauses,
    check_vector_column,
    filter_vector_columns,
)

logger = logging.getLogger(__name__)


class Table:
    def __init__(
        self,
        *,
        db_engine: Engine,
        schema: Optional[TableModel] = None,
        vector_column: Optional[str] = None,
        distance_metric: Optional[DistanceMetric] = DistanceMetric.COSINE,
        embed_model: Optional[EmbeddingModel] = None,
    ):
        self._db_engine = db_engine
        self._embed_model = embed_model

        # Init table model.
        if type(schema) is SQLModelMetaclass:
            self._table_model = schema
        elif type(schema) is DeclarativeMeta:
            self._table_model = schema
        else:
            raise TypeError(f"Invalid schema type: {type(schema)}")
        self._columns = self._table_model.__table__.columns

        # Create table.
        Base.metadata.create_all(self._db_engine, tables=[self._table_model.__table__])

        # Create index.
        self._vector_columns = filter_vector_columns(self._columns)
        vector_adaptor = VectorAdaptor(self._db_engine)
        for col in self._vector_columns:
            if vector_adaptor.has_vector_index(col):
                continue
            vector_adaptor.create_vector_index(col, distance_metric)

        # Determine vector column for search.
        if vector_column is not None:
            self._vector_column = check_vector_column(self._columns, vector_column)
        else:
            if len(self._vector_columns) == 1:
                self._vector_column = self._vector_columns[0]
            else:
                self._vector_column = None

    @property
    def table_model(self) -> TableModel:
        return self._table_model

    @property
    def table_name(self) -> str:
        return self._table_model.__tablename__

    @property
    def db_engine(self) -> Engine:
        return self._db_engine

    @property
    def vector_column(self):
        return self._vector_column

    @property
    def vector_columns(self):
        return self._vector_columns

    def get(self, id: int):
        with Session(self._db_engine) as session:
            return session.get(self._table_model, id)

    def insert(self, obj: object):
        # Auto embedding.
        # for vector_column in self._vector_columns:
        #     if self._embed_model is None:
        #         raise ValueError("please provide embed_model when table created")
        #
        #     embedding = self._embed_model.get_source_embedding(obj[self._vector_column.name])
        #     obj[vector_column.name] = embedding

        with Session(self._db_engine) as session:
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj

    def bulk_insert(self, objs: List[object]) -> List[object]:
        # Auto embedding.
        # for vector_column in self._vector_columns:
        #     obj_need_embedding = []
        #     texts_need_embedding = []
        #     for obj in objs:
        #         if obj[vector_column.name] is None:
        #             obj_need_embedding.append(None)
        #             texts_need_embedding.append(obj[self._source_column.name])
        #
        #     if self._embed_model is None:
        #         raise ValueError("please provide embed_model when table created")
        #
        #     embeddings = self._embed_model.get_source_embedding_batch(texts_need_embedding)
        #     for (obj, embedding) in zip(obj_need_embedding, embeddings):
        #         obj[self._vector_column.name] = embedding

        with Session(self._db_engine) as session:
            session.add_all(objs)
            session.commit()
            for obj in objs:
                session.refresh(obj)
            return objs

    def update(self, values: dict, filters: Optional[Dict[str, Any]] = None) -> object:
        filter_clauses = build_filter_clauses(filters, self._columns, self._table_model)
        with Session(self._db_engine) as session:
            stmt = update(self._table_model).filter(*filter_clauses).values(values)
            session.execute(stmt)
            session.commit()

    def delete(self, filters: Optional[Dict[str, Any]] = None):
        """
        Delete data from the TiDB table.

        params:
            filters: (Optional[Dict[str, Any]]): The filters to apply to the delete operation.
        """
        filter_clauses = build_filter_clauses(filters, self._columns, self._table_model)
        with Session(self._db_engine) as session:
            stmt = sqlalchemy.delete(self._table_model).filter(*filter_clauses)
            session.execute(stmt)
            session.commit()

    def truncate(self):
        with Session(self._db_engine) as session:
            stmt = text(f"TRUNCATE TABLE {self.table_name}")
            session.execute(stmt)

    def rows(self):
        with Session(self._db_engine) as session:
            stmt = text(f"SELECT COUNT(*) FROM {self.table_name}")
            res = session.execute(stmt)
            return res.scalar()

    def query(self, filters: Optional[Dict[str, Any]] = None):
        with Session(self._db_engine) as session:
            query = session.query(self._table_model)
            if filters:
                filter_clauses = build_filter_clauses(
                    filters, self._columns, self._table_model
                )
                query = query.filter(*filter_clauses)
            return query.all()

    def search(
        self,
        query: VectorDataType | str,
        query_type: QueryType = QueryType.VECTOR_SEARCH,
    ):
        if query_type == QueryType.VECTOR_SEARCH:
            # Auto embedding
            # if isinstance(query, str):
            #     if self._embed_model is None:
            #         raise ValueError("query is a string, please provide embed_model when table created")
            #     else:
            #         query = self._embed_model.get_query_embedding(query)

            return TiDBVectorQuery(
                table=self,
                query=query,
            )
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
