import logging
from typing import Any, List

import llama_index.core.instrumentation as instrument
from sqlmodel import SQLModel
from autoflow.storage.doc_store import DocumentStore
from autoflow.schema import BaseComponent

logger = logging.getLogger(__name__)


dispatcher = instrument.get_dispatcher(__name__)


class VectorSearchIndex(BaseComponent):
    def __init__(
        self,
        doc_store: DocumentStore,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._doc_store = doc_store

    def build_index_for_chunks(self, chunks: List[SQLModel]):
        if len(chunks) == 0:
            return chunks

        self._doc_store.add_doc_chunks(chunks)
