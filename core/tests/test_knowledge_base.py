import logging
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from autoflow.schema import DataSourceType, IndexMethod
from autoflow.main import Autoflow
from autoflow.llms import (
    ChatModel,
    EmbeddingModel,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def af():
    db_engine = create_engine(os.getenv("DATABASE_URL"))
    return Autoflow(db_engine)


@pytest.fixture(scope="module")
def chat_model():
    return ChatModel(
        "openai/gpt-4o-mini",
    )


@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel(model_name="openai/text-embedding-3-small", dimensions=1536)


@pytest.fixture(scope="module")
def test_kb(af, chat_model, embedding_model):
    kb = af.create_knowledge_base(
        id=uuid.UUID("438476e0-74f2-480f-b220-573e3f663d52"),
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.VECTOR_SEARCH, IndexMethod.KNOWLEDGE_GRAPH],
        chat_model=chat_model,
        embedding_model=embedding_model,
    )
    logger.info("Created knowledge base <%s> successfully.", kb.id)
    return kb


def test_import_documents_from_files(test_kb):
    test_kb.import_documents_from_files(
        files=[
            Path(__file__).parent / "fixtures" / "analyze-slow-queries.md",
            Path(__file__).parent / "fixtures" / "tidb-overview.md",
        ]
    )


def test_import_documents_from_datasource(test_kb):
    ds = test_kb.import_documents_from_datasource(
        type=DataSourceType.WEB_SINGLE_PAGE,
        config={"urls": ["https://docs.pingcap.com/tidbcloud/tidb-cloud-intro"]},
    )
    assert ds.id is not None


def test_search_documents(test_kb):
    result = test_kb.search_documents(
        query="What is TiDB?",
        similarity_top_k=2,
    )
    assert len(result.chunks) > 0


def test_search_knowledge_graph(test_kb):
    knowledge_graph = test_kb.search_knowledge_graph(
        query="What is TiDB?",
    )
    assert len(knowledge_graph.entities) > 0
    assert len(knowledge_graph.relationships) > 0
