from enum import Enum
import os
from typing import IO, Optional, Union, BinaryIO, TextIO
from urllib.parse import urlparse


class DataType(str, Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    SITEMAP = "sitemap"
    HTML = "html"


def guess_datatype(source: Union[str, IO, BinaryIO, TextIO]) -> Optional[DataType]:
    if isinstance(source, str):
        url = urlparse(source)
        if url.scheme == "" or url.scheme == "file":
            return guess_by_filename(url.path)
        elif url.scheme == "http" or url.scheme == "https":
            return DataType.HTML
        else:
            if os.path.exists(source):
                return guess_by_filename(source)
            raise ValueError(f"Unsupported URL scheme: {url.scheme}")
    elif isinstance(source, IO):
        return guess_by_filename(source.name)
    else:
        return None


def guess_by_filename(filename: str) -> Optional[DataType]:
    """Helper function to guess data type from filename."""
    lower = filename.lower()
    if lower.endswith(".md"):
        return DataType.MARKDOWN
    elif lower.endswith(".pdf"):
        return DataType.PDF
    elif lower.endswith(".docx"):
        return DataType.DOCX
    elif lower.endswith(".pptx"):
        return DataType.PPTX
    elif lower.endswith(".xlsx"):
        return DataType.XLSX
    elif lower.endswith(".csv"):
        return DataType.CSV
    elif lower.endswith(".xml") and "sitemap" in lower:
        return DataType.SITEMAP
    elif lower.endswith((".html", ".htm")):
        return DataType.HTML
    else:
        return None
