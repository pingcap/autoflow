import logging
import os

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.schema import NodeWithScore, TextNode
import pytest

from autoflow.configs.models.providers import ModelProviders
from autoflow.configs.models.providers.openai import OpenAIConfig
from autoflow.models.manager import (
    model_manager,
    ProviderConfig,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def setup_model_manager():
    model_manager.registry_provider(
        name=ModelProviders.OPENAI,
        config=OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
    )

    model_manager.registry_provider(
        name=ModelProviders.JINA_AI,
        config=ProviderConfig(
            api_key=os.getenv("JINAAI_API_KEY"),
        ),
    )


def test_llm():
    llm = model_manager.resolve_llm(
        provider=ModelProviders.OPENAI,
        config={
            "model": "gpt-4o",
        },
    )

    res = llm.chat(
        messages=[
            ChatMessage(
                role="user",
                content="Does TiDB Support Vector Search (Y/N)?",
            )
        ],
        max_tokens=1,
    )
    assert res.message.content is not None
    logger.info(
        f"LLM Answer: {res.message.content}",
    )


def test_embedding_model():
    embed_model = model_manager.resolve_embedding_model(
        provider=ModelProviders.OPENAI,
        config={
            "model": "text-embedding-3-small",
            "dimensions": 1536,
        },
    )
    vector = embed_model.get_query_embedding("What is TiDB?")
    assert len(vector) == 1536


def test_reranker_model():
    reranker_model = model_manager.resolve_rerank_model(
        provider=ModelProviders.JINA_AI,
        config={"model": "jina-reranker-v2-base-multilingual"},
    )
    nodes = reranker_model.postprocess_nodes(
        query_str="Database",
        nodes=[
            NodeWithScore(node=TextNode(text="Redis")),
            NodeWithScore(node=TextNode(text="OpenAI")),
            NodeWithScore(node=TextNode(text="TiDB")),
        ],
    )
    assert len(nodes) == 3
