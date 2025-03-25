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
    TypeVar,
    Generic,
)
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import asc, desc, select, and_, Result
from sqlalchemy.orm import Session
from autoflow.storage.tidb.schema import DistanceMetric, VectorDataType, TableModel
from autoflow.storage.tidb.utils import build_filter_clauses, check_vector_column


if TYPE_CHECKING:
    from autoflow.storage.tidb.table import Table
    from pandas import DataFrame


class SearchType(str, Enum):
    VECTOR_SEARCH = "vector_search"
    FULLTEXT_SEARCH = "fulltext_search"
    HYBRID_SEARCH = "hybrid_search"


class SearchQuery(ABC):
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
        query_type: SearchType,
    ) -> "SearchQuery":
        if query_type == SearchType.VECTOR_SEARCH:
            return VectorSearchQuery(
                table=table,
                query=query,
            )
        elif query_type == SearchType.FULLTEXT_SEARCH:
            raise NotImplementedError
        elif query_type == SearchType.HYBRID_SEARCH:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def limit(self, limit: Union[int, None]) -> "SearchQuery":
        if limit is None or limit <= 0:
            if isinstance(self, VectorSearchQuery):
                raise ValueError("Limit is required for ANN/KNN queries")
            else:
                self._limit = None
        else:
            self._limit = limit
        return self

    @abstractmethod
    def _execute(self, *args, **kwargs) -> Result:
        raise NotImplementedError

    @abstractmethod
    def to_rows(self) -> Sequence[Any]:
        raise NotImplementedError()

    @abstractmethod
    def to_list(self) -> List[dict]:
        raise NotImplementedError()

    @abstractmethod
    def to_pandas(self) -> Sequence[Any]:
        raise NotImplementedError()

    @abstractmethod
    def to_pydantic(self) -> "DataFrame":
        raise NotImplementedError()


SIMILARITY_SCORE_LABEL = "similarity_score"
SCORE_LABEL = "score"


class VectorSearchQuery(SearchQuery):
    def __init__(self, table: "Table", query: VectorDataType):
        super().__init__(table)
        if self._limit is None:
            self._limit = 10
        self._query = query
        self._distance_metric = DistanceMetric.COSINE
        self._distance_threshold = None
        self._distance_lower_bound = None
        self._distance_upper_bound = None
        self._num_candidate = 20
        self._vector_column = table.vector_column
        self._filters = None

    def vector_column(self, column_name: str):
        self._vector_column = check_vector_column(self._columns, column_name)
        return self

    def distance_metric(self, metric: DistanceMetric) -> "VectorSearchQuery":
        self._distance_metric = metric
        return self

    def distance_range(
        self, lower_bound: float = 0, upper_bound: float = 1
    ) -> "VectorSearchQuery":
        self._distance_lower_bound = lower_bound
        self._distance_upper_bound = upper_bound
        return self

    def num_candidate(self, num_candidate: int) -> "VectorSearchQuery":
        self._num_candidate = num_candidate
        return self

    def filter(self, filters: Optional[Dict[str, Any]] = None) -> "VectorSearchQuery":
        self._filters = filters
        return self

    def limit(self, k: int) -> "VectorSearchQuery":
        self._limit = k
        return self

    def _execute(self, db_session: Session) -> Result:
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
        subquery_stmt = (
            select(columns, distance_column)
            .order_by(asc(distance_label))
            .limit(num_candidate)
        )

        # Distance range.
        having = []
        if self._distance_lower_bound and self._distance_upper_bound:
            having.append(distance_column >= self._distance_lower_bound)
            having.append(distance_column <= self._distance_upper_bound)

        # Distance threshold.
        if self._distance_threshold:
            having.append(distance_column >= self._distance_threshold)

        if len(having) > 0:
            subquery_stmt = subquery_stmt.having(and_(*having))

        subquery = subquery_stmt.subquery("candidates")

        # Main query with metadata filters
        query = select(
            subquery.c,
            (1 - subquery.c[distance_label]).label(SIMILARITY_SCORE_LABEL),
            (1 - subquery.c[distance_label]).label(SCORE_LABEL),
        )

        if self._filters is not None:
            filter_clauses = build_filter_clauses(
                self._filters, subquery.c, table_model
            )
            query = query.filter(*filter_clauses)

        query = query.order_by(desc(SIMILARITY_SCORE_LABEL)).limit(self._limit)

        sql = query.compile(dialect=db_engine.dialect)
        logging.info(sql)

        return db_session.execute(query)

    def to_rows(self) -> Sequence[Any]:
        with Session(self._table.db_engine) as db_session:
            result = self._execute(db_session)
            return result.fetchall()

    def to_list(self) -> List[dict]:
        with Session(self._table.db_engine) as db_session:
            res = self._execute(db_session)
            keys = res.keys()
            rows = res.fetchall()
            return [dict(zip(keys, row)) for row in rows]

    def to_pydantic(self, with_score: Optional[bool] = True) -> List[BaseModel]:
        table_model = self._table.table_model

        with Session(self._table.db_engine) as db_session:
            result = self._execute(db_session)
            rows = result.fetchall()
            results = []
            for row in rows:
                values: Dict[str, Any] = dict(row._mapping)
                similarity_score: float = values.pop(SIMILARITY_SCORE_LABEL)
                score: float = values.pop(SCORE_LABEL)
                hit = table_model.model_validate(values)

                if not with_score:
                    results.append(hit)
                else:
                    results.append(
                        SearchResultModel(
                            similarity_score=similarity_score,
                            score=score,
                            hit=hit,
                        )
                    )

        return results

    def to_pandas(self) -> "DataFrame":
        try:
            import pandas as pd
        except Exception:
            raise ImportError(
                "Failed to import pandas, please install it with `pip install pandas`"
            )

        with Session(self._table.db_engine) as db_session:
            result = self._execute(db_session)
            keys = result.keys()
            rows = result.fetchall()
            return pd.DataFrame(rows, columns=keys)


T = TypeVar("T", bound=TableModel)


class SearchResultModel(BaseModel, Generic[T]):
    hit: T
    score: Optional[float] = Field(None)
    similarity_score: Optional[float] = Field(None)

    def __getattr__(self, item: str):
        if hasattr(self.hit, item):
            return getattr(self.hit, item)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )
