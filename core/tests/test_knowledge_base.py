import logging
import os
import uuid
from pathlib import Path

from sqlalchemy import create_engine

from autoflow.schema import DataSourceType, IndexMethod
from autoflow.main import Autoflow
from autoflow.llms import (
    ChatModel,
    EmbeddingModel,
)

logger = logging.getLogger(__name__)

db_engine = create_engine(os.getenv("DATABASE_URL"))
af = Autoflow(db_engine)

chat_model = ChatModel(
    "openai/gpt-4o-mini",
)
embedding_model = EmbeddingModel(
    model_name="openai/text-embedding-3-small", dimensions=1536
)


def test_create_knowledge_base():
    kb = af.create_knowledge_base(
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.VECTOR_SEARCH, IndexMethod.KNOWLEDGE_GRAPH],
        chat_model=chat_model,
        embedding_model=embedding_model,
    )
    logger.info("Created knowledge base <%s> successfully.", kb.id)


def test_import_documents_from_files():
    kb = af.create_knowledge_base(
        kb_id=uuid.UUID("01973588-65aa-4954-99fd-71eb5ecce167"),
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.VECTOR_SEARCH, IndexMethod.KNOWLEDGE_GRAPH],
        chat_model=chat_model,
        embedding_model=embedding_model,
    )

    kb.import_documents_from_files(
        files=[
            Path(__file__).parent / "fixtures" / "analyze-slow-queries.md",
            Path(__file__).parent / "fixtures" / "tidb-overview.md",
        ]
    )


def test_import_documents_from_datasource():
    kb = af.create_knowledge_base(
        kb_id=uuid.UUID("01973588-65aa-4954-99fd-71eb5ecce167"),
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.VECTOR_SEARCH],
        chat_model=chat_model,
        embedding_model=embedding_model,
    )
    ds = kb.import_documents_from_datasource(
        type=DataSourceType.WEB_SINGLE_PAGE,
        config={"urls": ["https://docs.pingcap.com/tidbcloud/tidb-cloud-intro"]},
    )
    logger.info("Created data source <%s> successfully.", ds.id)


def test_search_documents():
    kb = af.create_knowledge_base(
        kb_id=uuid.UUID("01973588-65aa-4954-99fd-71eb5ecce167"),
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.VECTOR_SEARCH],
        chat_model=chat_model,
        embedding_model=embedding_model,
    )

    result = kb.search_documents(
        query="What is TiDB?",
        similarity_top_k=2,
    )
    assert len(result.chunks) == 2


def test_search_knowledge_graph():
    kb = af.create_knowledge_base(
        kb_id=uuid.UUID("01973588-65aa-4954-99fd-71eb5ecce167"),
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.KNOWLEDGE_GRAPH],
        chat_model=chat_model,
        embedding_model=embedding_model,
    )

    kg = kb.search_knowledge_graph(
        query="What is TiDB?",
    )
    assert len(kg.entities) > 0
    assert len(kg.relationships) > 0
