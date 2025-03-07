import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Sequence,
)

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import asc, desc, Row, select
from sqlalchemy.orm import Session

from autoflow.storage.tidb.constants import DistanceMetric, VectorDataType
from autoflow.storage.tidb.utils import build_filter_clauses, check_vector_column

if TYPE_CHECKING:
    from autoflow.storage.tidb.table import Table


class Query(BaseModel):
    vector_column: Optional[str] = None
    vector: Union[List[float], List[List[float]]]
    filter: Optional[Dict[str, Any]] = None
    distance_metric: str = "L2"
    distance_range: Optional[Tuple[float, float]] = None
    nprobes: int = 10
    top_k: Optional[int] = None
    use_index: bool = True


class QueryType(str, Enum):
    VECTOR_SEARCH = "vector_search"
    FULLTEXT_SEARCH = "fulltext_search"
    HYBRID_SEARCH = "hybrid_search"


class TiDBQuery(ABC):
    def __init__(self, table: "Table"):
        self._table = table
        self._limit = None
        self._offset = 0
        self._columns = None
        self._where = None
        self._prefilter = True
        self._with_row_id = False
        self._vector = None
        self._text = None
        self._ef = None
        self._use_index = True

    @classmethod
    def create(
        cls,
        table: "Table",
        query: Optional[Union[np.ndarray, str, Tuple]],
        query_type: QueryType,
    ) -> "TiDBQuery":
        if query_type == QueryType.VECTOR_SEARCH:
            return TiDBVectorQuery(
                table=table,
                query=query,
            )
        elif query_type == QueryType.FULLTEXT_SEARCH:
            raise NotImplementedError
        elif query_type == QueryType.HYBRID_SEARCH:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def limit(self, limit: Union[int, None]) -> "TiDBQuery":
        if limit is None or limit <= 0:
            if isinstance(self, TiDBVectorQuery):
                raise ValueError("Limit is required for ANN/KNN queries")
            else:
                self._limit = None
        else:
            self._limit = limit
        return self

    def execute(self):
        return self._execute()

    @abstractmethod
    def _execute(self):
        pass

    def to_rows(self) -> Sequence[Row]:
        return self.execute()

    def to_pandas(self) -> pd.DataFrame:
        raise NotImplementedError


class TiDBVectorQuery(TiDBQuery):
    def __init__(self, table: "Table", query: VectorDataType):
        super().__init__(table)
        if self._limit is None:
            self._limit = 10
        self._query = query
        self._distance_metric = DistanceMetric.COSINE
        self._num_candidate = 20
        self._vector_column = table.vector_column
        self._filters = None

    def vector_column(self, column_name: str):
        self._vector_column = check_vector_column(self._columns, column_name)
        return self

    def distance_metric(self, metric: DistanceMetric) -> "TiDBVectorQuery":
        self._distance_metric = metric
        return self

    def num_candidate(self, num_candidate: int) -> "TiDBVectorQuery":
        self._num_candidate = num_candidate
        return self

    def filter(self, filters: Optional[Dict[str, Any]] = None) -> "TiDBVectorQuery":
        self._filters = filters
        return self

    def limit(self, k: int) -> "TiDBVectorQuery":
        self._limit = k
        return self

    def _execute(self) -> Sequence[Row]:
        num_candidate = self._num_candidate if self._num_candidate else self._limit * 10

        if self._vector_column is None:
            if len(self._table.vector_columns) == 0:
                raise ValueError(
                    "no vector column found in the table, vector search cannot be executed"
                )
            elif len(self._table.vector_columns) >= 1:
                raise ValueError(
                    "more than two vector columns in the table, need to be specified one through .vector_column()"
                )
            else:
                vector_column = self._table.vector_columns[0]
        else:
            vector_column = self._vector_column

        # Auto embedding
        if isinstance(self._query, str):
            if vector_column.name not in self._table.vector_field_configs:
                raise ValueError()

            config = self._table.vector_field_configs[vector_column.name]
            self._query = config["embed_fn"].get_query_embedding(self._query)

        # Distance metric.
        distance_label = "_distance"
        if self._distance_metric == DistanceMetric.L2:
            distance_column = vector_column.l2_distance(self._query).label(
                distance_label
            )
        else:
            distance_column = vector_column.cosine_distance(self._query).label(
                distance_label
            )

        # Inner query for ANN search
        db_engine = self._table.db_engine
        table_model = self._table.table_model
        columns = table_model.__table__.c
        subquery = (
            select(columns, distance_column)
            .order_by(asc(distance_label))
            .limit(num_candidate)
            .subquery("candidates")
        )

        # Main query with metadata filters
        query = select(
            subquery.c,
            (1 - subquery.c[distance_label]).label("similarity_score"),
        )

        if self._filters is not None:
            filter_clauses = build_filter_clauses(
                self._filters, subquery.c, table_model
            )
            query = query.filter(*filter_clauses)

        query = query.order_by(desc("similarity_score")).limit(self._limit)

        sql = query.compile(dialect=db_engine.dialect)
        logging.info(sql)

        with Session(db_engine) as session:
            return session.execute(query).all()

    def to_pydantic(self) -> List[BaseModel]:
        rows = self.execute()
        results = []
        for row in rows:
            model = self._table.table_model.model_validate(row._mapping)
            if row._mapping["similarity_score"] is not None:
                model.similarity_score = row._mapping["similarity_score"]
                model.score = row._mapping["similarity_score"]
            results.append(model)
        return results
