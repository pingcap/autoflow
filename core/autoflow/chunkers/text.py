from typing import Optional

from autoflow.chunkers.base import Chunker
from autoflow.configs.chunkers.text import TextChunkerConfig
from autoflow.storage.doc_store import Document, Chunk


class TextChunker(Chunker):
    """Chunker for text."""

    def __init__(self, config: Optional[TextChunkerConfig] = TextChunkerConfig()):
        super().__init__()
        from llama_index.core.node_parser import SentenceSplitter

        self._splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def chunk(self, document: Document) -> Document:
        texts = self._splitter.split_text(document.content)
        document.chunks = [Chunk(text=text) for text in texts]
        return document
