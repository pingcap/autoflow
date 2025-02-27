from pathlib import Path
from typing import List, Optional, Dict, Any, Generator

from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import Engine
from sqlalchemy.orm.decl_api import RegistryType
from sqlmodel import Session
from sqlmodel.main import default_registry

from autoflow.datasources import DataSource
from autoflow.db_models.chunk import get_chunk_model
from autoflow.db_models import DataSourceKind
from autoflow.db_models.entity import get_entity_model
from autoflow.db_models.relationship import get_relationship_model
from autoflow.knowledge_base.config import (
    IndexMethod,
    LLMConfig,
    EmbeddingModelConfig,
    DEFAULT_INDEX_METHODS,
)
from autoflow.db_models.document import Document
from autoflow.knowledge_base.datasource import get_datasource_by_kind
from autoflow.models import default_model_manager, ModelManager
from autoflow.stores import TiDBDocumentStore, TiDBKnowledgeGraphStore
from autoflow.db_models import DBKnowledgeBase
from autoflow.stores.document_store.base import DocumentSearchResult


class KnowledgeBase:
    _kb: Optional[DBKnowledgeBase] = None
    _registry: RegistryType

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        index_methods: Optional[List[IndexMethod]] = None,
        llm: LLMConfig = None,
        embedding_model: EmbeddingModelConfig = None,
        db_engine: Engine = None,
        model_manager: Optional[ModelManager] = None,
        kb: Optional[DBKnowledgeBase] = None,
    ):
        self._db_engine = db_engine
        if model_manager is None:
            model_manager = default_model_manager

        if kb is None:
            with Session(db_engine) as db_session:
                kb = DBKnowledgeBase(
                    name=name,
                    description=description,
                    index_methods=index_methods or DEFAULT_INDEX_METHODS,
                    llm=llm,
                    embedding_model=embedding_model,
                )
                db_session.add(kb)
                db_session.commit()
                db_session.refresh(kb)

        self._kb = kb
        self._llm = model_manager.resolve_llm(llm)
        self._embedding_model = model_manager.resolve_embedding_model(embedding_model)
        self._init_stores()

    def _init_stores(self):
        namespace_id = f"{self._kb.id}"
        vector_dimension = self._embedding_model.dimensions

        self._registry = RegistryType(
            metadata=default_registry.metadata,
            class_registry=default_registry._class_registry.copy(),
        )

        # Init chunk table.
        document_table_name = "documents"
        chunk_table_name = f"chunks_{namespace_id}"
        self._chunk_db_model = get_chunk_model(
            chunk_table_name,
            vector_dimension=vector_dimension,
            document_table_name=document_table_name,
            document_db_model=Document,
            registry=self._registry,
        )

        # Init entity table.
        entity_table_name = f"entities_{namespace_id}"
        self._entity_db_model = get_entity_model(
            entity_table_name,
            vector_dimension=vector_dimension,
            registry=self._registry,
        )

        # Init relationship table.
        relationship_table_name = f"relationships_{namespace_id}"
        self._relationship_db_model = get_relationship_model(
            relationship_table_name,
            vector_dimension=vector_dimension,
            entity_db_model=self._entity_db_model,
            registry=self._registry,
        )

        self._doc_store = TiDBDocumentStore(
            db_engine=self._db_engine,
            embedding_model=self._embedding_model,
            document_db_model=Document,
            chunk_db_model=self._chunk_db_model,
        )

        self._graph_store = TiDBKnowledgeGraphStore(
            db_engine=self._db_engine,
            embedding_model=self._embedding_model,
            entity_db_model=self._entity_db_model,
            relationship_db_model=self._relationship_db_model,
        )

    @property
    def id(self) -> int:
        return self._kb.id

    @property
    def name(self):
        return self._kb.name

    @property
    def description(self):
        return self._kb.description

    @property
    def index_methods(self):
        return self._kb.index_methods

    @property
    def registry(self):
        return self._registry

    @property
    def llm(self):
        return self._llm

    @property
    def embedding_model(self):
        return self._embedding_model

    def create_datasource(
        self,
        kind: DataSourceKind,
        name: str,
        config: Dict[str, Any],
        load_documents: bool = True,
        build_index: bool = True,
        # TODO: Metadata Extractor
    ) -> DataSource:
        with Session(self._db_engine) as db_session:
            ds = DataSource(kind=kind, name=name, config=config)
            db_session.add(ds)
            self._kb.datasources.append(ds)
            db_session.commit()
        datasource = get_datasource_by_kind(kind, config)

        if load_documents:
            for doc in datasource.load_documents():
                if build_index:
                    self.build_index_for_document(doc)

        return datasource

    def build_index_for_datasource(self, doc: Document):
        pass

    def delete_datasource(self, datasource_id: int):
        with Session(self._db_engine) as db_session:
            self._kb.datasources.remove(datasource_id)
            # TODO: Remove documents.
            db_session.commit()

    def list_datasources(self) -> List[DataSource]:
        with Session(self._db_engine, expire_on_commit=False) as db_session:
            db_session.refresh(self._kb)
            return [DataSource.from_db(ds) for ds in self._kb.datasources]

    def import_document_from_file(
        self, file: Path, build_index: bool = True
    ) -> List[Document]:
        data_source = self.create_datasource(
            kind=DataSourceKind.FILE,
            name=f"Local File: {file.name}",
            config={"files": [{"file": file.as_uri()}]},
        )
        documents = []
        for doc in data_source.load_documents():
            if build_index:
                documents.append(self.build_index_for_document(doc))
            else:
                documents.append(doc)
        return documents

    def add_document(self, document: Document):
        return self._doc_store.add([document])

    def add_documents(self, documents: List[Document]):
        return self._doc_store.add(documents)

    def list_documents(self) -> List[Document]:
        return self._doc_store.list()

    def get_document(self, doc_id: int) -> Document:
        return self._doc_store.get(doc_id)

    def delete_document(self, doc_id: int) -> None:
        return self._doc_store.delete(doc_id)

    def build_index_for_document(self, document: Document):
        pass

    def _document_pipeline(self, documents: List[Document]):
        pass

    def _document_text_split(
        self, documents: Generator[Document, Any, None]
    ) -> Generator[Document, Any, None]:
        # TODO: Support more text splitter.
        text_splitter = SentenceSplitter()
        for doc in documents:
            chunk_texts = text_splitter.split_text(doc.content)
            chunks = [
                self._chunk_db_model(
                    text=chunk_text,
                )
                for chunk_text in chunk_texts
            ]
            doc.chunks = chunks
            yield doc

    def _document_build_index(
        self, documents: Generator[Document, Any, None]
    ) -> Generator[Document, Any, None]:
        pass
        # extractor = SimpleGraphExtractor(dspy_lm=self._dspy_lm)
        # for node in nodes:
        #     entities_df, rel_df = extractor.extract(
        #         text=node.get_content(),
        #         node=node,
        #     )

    def _build_vector_search_index(self, document: Document):
        pass

    def _build_knowledge_graph_index(self, document: Document):
        pass

    def search_document(self, **kwargs) -> DocumentSearchResult:
        return self._doc_store.search(**kwargs)
