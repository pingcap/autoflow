import logging
import os
import uuid
from pathlib import Path

from sqlalchemy import create_engine

from autoflow.schema import DataSourceKind, IndexMethod
from autoflow.main import Autoflow
from autoflow.llms import (
    EmbeddingModelConfig,
    LLMProviders,
    ProviderConfig,
    ChatModelConfig,
)
from autoflow.storage.doc_store import DocumentSearchQuery

logger = logging.getLogger(__name__)

db_engine = create_engine(os.getenv("DATABASE_URL"))
af = Autoflow(db_engine)
af.model_manager.configure_provider(
    name=LLMProviders.OPENAI,
    config=ProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
)


def test_create_knowledge_base():
    kb = af.crate_knowledge_base(
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.VECTOR_SEARCH, IndexMethod.KNOWLEDGE_GRAPH],
        chat_model=ChatModelConfig(provider=LLMProviders.OPENAI, model="gpt4o-mini"),
        embedding_model=EmbeddingModelConfig(
            provider=LLMProviders.OPENAI,
            model="text-embedding-3-small",
            dimensions=1536,
        ),
    )
    logger.info("Created knowledge base #%d successfully.", kb.id)


def test_import_documents_from_files():
    kb = af.crate_knowledge_base(
        kb_id=uuid.UUID("01973588-65aa-4954-99fd-71eb5ecce167"),
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.VECTOR_SEARCH, IndexMethod.KNOWLEDGE_GRAPH],
        chat_model=ChatModelConfig(provider=LLMProviders.OPENAI, model="gpt-4o-mini"),
        embedding_model=EmbeddingModelConfig(
            provider=LLMProviders.OPENAI,
            model="text-embedding-3-small",
            dimensions=1536,
        ),
    )

    kb.import_documents_from_files(
        files=[
            Path(__file__).parent / "fixtures" / "analyze-slow-queries.md",
        ]
    )


def test_import_documents_from_datasource():
    kb = af.crate_knowledge_base(
        kb_id=uuid.UUID("01973588-65aa-4954-99fd-71eb5ecce167"),
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.VECTOR_SEARCH],
        chat_model=ChatModelConfig(provider=LLMProviders.OPENAI, model="gpt-4o-mini"),
        embedding_model=EmbeddingModelConfig(
            provider=LLMProviders.OPENAI,
            model="text-embedding-3-small",
            dimensions=1536,
        ),
    )
    ds = kb.import_documents_from_datasource(
        kind=DataSourceKind.WEB_SINGLE_PAGE,
        config={"urls": ["https://docs.pingcap.com/tidbcloud/tidb-cloud-intro"]},
    )
    logger.info("Created data source #%s successfully.", ds.id)


def test_search_documents():
    kb = af.crate_knowledge_base(
        kb_id=uuid.UUID("01973588-65aa-4954-99fd-71eb5ecce167"),
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.VECTOR_SEARCH],
        chat_model=ChatModelConfig(provider=LLMProviders.OPENAI, model="gpt-4o-mini"),
        embedding_model=EmbeddingModelConfig(
            provider=LLMProviders.OPENAI,
            model="text-embedding-3-small",
            dimensions=1536,
        ),
    )

    kb.import_documents_from_files(
        files=[
            Path(__file__).parent / "fixtures" / "analyze-slow-queries.md",
            Path(__file__).parent / "fixtures" / "tidb-overview.md",
        ]
    )

    result = kb.search_documents(
        DocumentSearchQuery(
            query_str="What is TiDB?",
            similarity_top_k=2,
        )
    )
    assert len(result.chunks) == 2


def test_search_knowledge_graph():
    kb = af.crate_knowledge_base(
        kb_id=uuid.UUID("01973588-65aa-4954-99fd-71eb5ecce167"),
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.KNOWLEDGE_GRAPH],
        chat_model=ChatModelConfig(provider=LLMProviders.OPENAI, model="gpt-4o-mini"),
        embedding_model=EmbeddingModelConfig(
            provider=LLMProviders.OPENAI,
            model="text-embedding-3-small",
            dimensions=1536,
        ),
    )

    # kb.import_documents_from_files(
    #     files=[
    #         Path(__file__).parent / "fixtures" / "analyze-slow-queries.md",
    #         Path(__file__).parent / "fixtures" / "tidb-overview.md",
    #     ]
    # )

    result = kb.search_knowledge_graph(
        query="What is TiDB?",
    )
    assert len(result.chunks) == 2


# def test_import_document_from_file():
#     test_kb_id = 1
