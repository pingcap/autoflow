from llama_index.readers import PDFReader

from autoflow.data_types import DataType
from autoflow.loaders.base import FileLoader
from autoflow.storage.doc_store import Document


class PDFLoader(FileLoader):
    def _load_file(self, file: str) -> Document:
        reader = PDFReader()
        documents = reader.load_data(file)
        content = documents[0].text

        return Document(
            name=file.name,
            data_type=DataType.PDF,
            content=content,
        )
