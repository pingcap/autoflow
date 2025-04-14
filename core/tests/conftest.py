import logging
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from pytidb import TiDBClient
from autoflow.models.llms import LLM
from autoflow.models.embedding_models import EmbeddingModel
from autoflow.configs.db import DatabaseConfig
from autoflow.db import get_db_engine_from_config

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def env():
    logger.info(f"Loading environment variables from {Path.cwd() / '.env'}")
    load_dotenv()


@pytest.fixture(scope="session")
def db_engine():
    config = DatabaseConfig(
        host=os.getenv("TIDB_HOST"),
        port=int(os.getenv("TIDB_PORT")),
        username=os.getenv("TIDB_USERNAME"),
        password=os.getenv("TIDB_PASSWORD"),
        database=os.getenv("TIDB_DATABASE"),
        enable_ssl=False,
    )
    return get_db_engine_from_config(config)


@pytest.fixture(scope="session")
def llm():
    return LLM(model="openai/gpt-4o-mini")


@pytest.fixture(scope="session")
def embedding_model():
    return EmbeddingModel(model_name="text-embedding-3-small")


@pytest.fixture(scope="session")
def tidb_client(db_engine):
    return TiDBClient(db_engine=db_engine)
