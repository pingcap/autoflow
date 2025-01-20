from typing import Mapping, Any
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from .metadata_post_filter import (
    MetadataFilters,
    MetadataPostFilter,
    MetadataFilter,
)


def get_metadata_post_filter(
    filters: Mapping[str, Any] = None,
) -> BaseNodePostprocessor:
    simple_filters = []
    for key, value in filters.items():
        simple_filters.append(
            MetadataFilter(
                key=key,
                value=value,
            )
        )
    return MetadataPostFilter(
        MetadataFilters(
            filters=simple_filters,
        )
    )
