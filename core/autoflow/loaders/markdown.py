from autoflow.data_types import DataType
from autoflow.loaders.base import FileLoader
from autoflow.storage.doc_store import Document


class MarkdownLoader(FileLoader):
    def _load_file(self, file: str) -> Document:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        return Document(
            name=file,
            data_type=DataType.MARKDOWN,
            content=content,
        )
