import logging

import pytest

from autoflow.configs.knowledge_base import IndexMethod
from autoflow.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def kb(db_engine, llm, embedding_model):
    kb = KnowledgeBase(
        namespace="test",
        name="Test",
        description="Here is a knowledge base with namespace",
        index_methods=[IndexMethod.VECTOR_SEARCH, IndexMethod.KNOWLEDGE_GRAPH],
        llm=llm,
        embedding_model=embedding_model,
        db_engine=db_engine,
    )
    logger.info(
        "Created a knowledge base with namespace %s successfully.", kb.namespace
    )
    return kb


def test_add_documents_via_filepath(kb: KnowledgeBase):
    docs = kb.add("./tests/fixtures/analyze-slow-queries.md")
    assert len(docs) == 1


def test_add_documents_via_url(kb):
    docs = kb.add("https://docs.pingcap.com/tidbcloud/tidb-cloud-intro")
    assert len(docs) == 1


def test_search_documents(kb):
    result = kb.search_documents(
        query="What is TiDB?",
        similarity_top_k=2,
    )
    assert len(result.chunks) > 0


def test_search_knowledge_graph(kb):
    knowledge_graph = kb.search_knowledge_graph(
        query="What is TiDB?",
    )
    assert len(knowledge_graph.entities) > 0
    assert len(knowledge_graph.relationships) > 0
