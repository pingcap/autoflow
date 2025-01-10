from typing import Optional
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from .metadata_post_filter import (
    MetadataFilters,
    MetadataPostFilter,
)


def get_metadata_post_filter(
    filters: Optional[MetadataFilters] = None,
) -> BaseNodePostprocessor:
    return MetadataPostFilter(filters)
