from autoflow.chunkers.base import Chunker
from autoflow.data_types import DataType


def get_chunker_for_datatype(datatype: DataType) -> Chunker:
    if datatype in [DataType.MARKDOWN, DataType.HTML, DataType.PDF]:
        from autoflow.chunkers.text import TextChunker

        return TextChunker()
    else:
        raise ValueError(f"Unsupported data type: {datatype}")
