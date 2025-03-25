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
