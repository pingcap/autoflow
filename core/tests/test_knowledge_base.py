import logging
import os

from sqlalchemy import create_engine

from autoflow.db_models import DataSourceKind
from autoflow.knowledge_base.config import IndexMethod
from autoflow.main import Autoflow
from autoflow.models import (
    EmbeddingModelConfig,
    LLMConfig,
    ModelProviders,
    ProviderConfig,
)

logger = logging.getLogger(__name__)

test_kb_id = 1
db_engine = create_engine(os.getenv("DATABASE_URL"))
af = Autoflow(db_engine)
af.model_manager.configure_provider(
    name=ModelProviders.OPENAI,
    config=ProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
)


def test_create_knowledge_base():
    kb = af.crate_knowledge_base(
        name="Test",
        description="This is a knowledge base for testing",
        index_methods=[IndexMethod.VECTOR_SEARCH, IndexMethod.KNOWLEDGE_GRAPH],
        llm=LLMConfig(provider=ModelProviders.OPENAI, model="gpt4o-mini"),
        embedding_model=EmbeddingModelConfig(
            provider=ModelProviders.OPENAI, model="text-embedding-3-small"
        ),
    )
    logger.info("Created knowledge base #%d successfully.", kb.id)


def test_crate_datasource():
    kb = af.get_knowledge_base(test_kb_id)
    ds = kb.create_datasource(
        kind=DataSourceKind.WEB_SINGLE_PAGE,
        name="TiDB Cloud Intro",
        config={"urls": ["https://docs.pingcap.com/tidbcloud/tidb-cloud-intro"]},
        load_documents=False,
        build_index=False,
    )
    logger.info("Created data source #%d successfully.", ds.id)


# def test_import_document_from_file():
#     test_kb_id = 1
