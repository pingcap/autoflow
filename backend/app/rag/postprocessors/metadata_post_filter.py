import logging
from enum import Enum
from typing import List, Optional, Any, Union

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel


class FilterOperator(str, Enum):
    """Vector store filter operator."""

    # TODO add more operators
    EQ = "=="  # default operator (string, int, float)
    GT = ">"  # greater than (int, float)
    LT = "<"  # less than (int, float)
    NE = "!="  # not equal to (string, int, float)
    GTE = ">="  # greater than or equal to (int, float)
    LTE = "<="  # less than or equal to (int, float)
    IN = "in"  # In array (string or number)
    NIN = "nin"  # Not in array (string or number)
    ANY = "any"  # Contains any (array of strings)
    ALL = "all"  # Contains all (array of strings)
    TEXT_MATCH = "text_match"  # full text match (allows you to search for a specific substring, token or phrase
    # within the text field)
    CONTAINS = "contains"  # metadata array contains value (string or number)


class FilterCondition(str, Enum):
    AND = "and"
    OR = "or"


class MetadataFilter(BaseModel):
    key: str
    value: Union[
        int,
        float,
        str,
        List[int],
        List[float],
        List[str],
    ]
    operator: FilterOperator = FilterOperator.EQ


# Notice:
#
# llama index is still heavily using pydantic v1 to define data models. Using classes in llama index to define FastAPI
# parameters may cause the following errors:
#
#   TypeError: BaseModel.validate() takes 2 positional arguments but 3 were given
#
# See: https://github.com/run-llama/llama_index/issues/14807#issuecomment-2241285940
class MetadataFilters(BaseModel):
    """Metadata filters for vector stores."""

    # Exact match filters and Advanced filters with operators like >, <, >=, <=, !=, etc.
    filters: List[Union[MetadataFilter, "MetadataFilters"]]
    # and/or such conditions for combining different filters
    condition: Optional[FilterCondition] = FilterCondition.AND


_logger = logging.getLogger(__name__)


class MetadataPostFilter(BaseNodePostprocessor):
    filters: Optional[MetadataFilters] = None

    def __init__(self, filters: Optional[MetadataFilters] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.filters = filters

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if self.filters is None:
            return nodes

        filtered_nodes = []
        for node in nodes:
            # TODO: support advanced post filtering.
            if self.match_all_filters(node.node):
                filtered_nodes.append(node)
        return filtered_nodes

    def match_all_filters(self, node: Any) -> bool:
        if self.filters is None or not isinstance(self.filters, MetadataFilters):
            return True

        if self.filters.condition != FilterCondition.AND:
            _logger.warning(
                f"Advanced filtering is not supported yet. "
                f"Filter condition {self.filters.condition} is ignored."
            )
            return True

        for f in self.filters.filters:
            if f.key not in node.extra_info:
                return False

            if f.operator is not None and f.operator != FilterOperator.EQ:
                _logger.warning(
                    f"Advanced filtering is not supported yet. "
                    f"Filter operator {f.operator} is ignored."
                )
                return True

            value = node.extra_info[f.key]
            if f.value != value:
                return False

        return True
