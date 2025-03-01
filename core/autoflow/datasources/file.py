import logging
import mimetypes
from pathlib import Path
from typing import Generator, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    PptxReader,
    PandasExcelReader,
    MarkdownReader,
    CSVReader,
)
from pydantic import BaseModel

from .base import DataSource, C
from .mime_types import SupportedMimeTypes
from autoflow.models import DBDocument

logger = logging.getLogger(__name__)


class FileConfig(BaseModel):
    path: str


class FileDataSourceConfigConfig(BaseModel):
    files: List[FileConfig]


def get_file_reader_by_mime_type(mime_type: str) -> Optional[BaseReader]:
    if mime_type == SupportedMimeTypes.PDF:
        return PDFReader(return_full_document=True)
    elif mime_type == SupportedMimeTypes.DOCX:
        return DocxReader()
    elif mime_type == SupportedMimeTypes.PPTX:
        return PptxReader()
    elif mime_type == SupportedMimeTypes.XLSX:
        return PandasExcelReader()
    elif mime_type == SupportedMimeTypes.CSV:
        return CSVReader()
    elif mime_type == SupportedMimeTypes.MARKDOWN:
        return MarkdownReader()
    else:
        return None


class FileDataSource(DataSource[FileDataSourceConfigConfig]):
    def validate_config(self, config: dict) -> C:
        return FileDataSourceConfigConfig.model_validate(config)

    def load_documents(self) -> Generator[DBDocument, None, None]:
        for fc in self.config.files:
            filepath = Path(fc.path)
            filename = filepath.name
            mime_type = mimetypes.guess_type(fc.path)[0]
            if mime_type is None:
                if fc.path.endswith(".md"):
                    mime_type = SupportedMimeTypes.MARKDOWN
            reader = get_file_reader_by_mime_type(mime_type)
            docs = reader.load_data(filepath)
            for doc in docs:
                document = DBDocument(
                    name=filename,
                    hash=hash(doc.text),
                    mime_type=SupportedMimeTypes.MARKDOWN
                    if mime_type == SupportedMimeTypes.MARKDOWN
                    else SupportedMimeTypes.PLAIN_TXT,
                    content=doc.text,
                )
                yield document
