import enum
import typing
import numpy
from sqlalchemy.orm import DeclarativeMeta
from sqlmodel.main import SQLModelMetaclass

# TiDB Vector has a limitation on the dimension length
MAX_DIM = 16000
MIN_DIM = 1

# Filter operators:

AND, OR, IN, NIN, GT, GTE, LT, LTE, EQ, NE = (
    "$and",
    "$or",
    "$in",
    "$nin",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$eq",
    "$ne",
)

COMPARE_OPERATOR = [IN, NIN, GT, GTE, LT, LTE, EQ, NE]

TableModel = typing.Union[SQLModelMetaclass, DeclarativeMeta]

VectorDataType = typing.Union[numpy.ndarray, typing.List[float]]


class DistanceMetric(enum.Enum):
    """
    An enumeration representing different types of distance metrics.

    - `DistanceMetric.L2`: L2 (Euclidean) distance metric.
    - `DistanceMetric.COSINE`: Cosine distance metric.
    """

    L2 = "L2"
    COSINE = "COSINE"

    def to_sql_func(self):
        """
        Converts the DistanceMetric to its corresponding SQL function name.

        Returns:
            str: The SQL function name.

        Raises:
            ValueError: If the DistanceMetric enum member is not supported.
        """
        if self == DistanceMetric.L2:
            return "VEC_L2_DISTANCE"
        elif self == DistanceMetric.COSINE:
            return "VEC_COSINE_DISTANCE"
        else:
            raise ValueError("unsupported distance metric")
