import logging

from typing import Dict, List, Optional, Any, Union
from app.rag.indices.vector_search.filters.filter_evaluator import FilterEvaluator
from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)


SimpleMetadataFilter = Dict[str, Any]


def simple_filter_to_metadata_filters(filters: SimpleMetadataFilter) -> MetadataFilters:
    simple_filters = []
    for key, value in filters.items():
        simple_filters.append(
            MetadataFilter(
                key=key,
                value=value,
                operator=FilterOperator.EQ,
            )
        )
    return MetadataFilters(filters=simple_filters)


logger = logging.getLogger(__name__)


class MetadataPostFilter(BaseNodePostprocessor):
    filters: Optional[MetadataFilters] = None

    def __init__(
        self,
        filters: Optional[Union[MetadataFilters, SimpleMetadataFilter]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if isinstance(filters, MetadataFilters):
            self.filters = filters
        else:
            self.filters = simple_filter_to_metadata_filters(filters)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if self.filters is None:
            return nodes

        filtered_nodes = []
        for node in nodes:
            if self.match_all_filters(node.node):
                filtered_nodes.append(node)
        return filtered_nodes

    def match_all_filters(self, node: BaseNode) -> bool:
        logger.debug(f"Process chunk with match_all_filters: {self.filters}")
        if self.filters is None or not isinstance(self.filters, MetadataFilters):
            return True


        if not self.filters.filters:
            return True
        
        return FilterEvaluator.evaluate_conditions(self.filters, metadata=node.metadata)