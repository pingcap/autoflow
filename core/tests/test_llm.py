import logging
import os

from llama_index.core.schema import NodeWithScore, TextNode
import pytest

from autoflow.llms import (
    default_llm_manager as llm_manager,
    ProviderConfig,
    LLMProviders,
    ChatModelConfig,
    EmbeddingModelConfig,
    RerankerModelConfig,
)
from autoflow.chat import ChatMessage

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def setup_llm_manager():
    llm_manager.configure_provider(
        name=LLMProviders.OPENAI,
        config=ProviderConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
    )

    llm_manager.configure_provider(
        name=LLMProviders.JINA_AI,
        config=ProviderConfig(
            api_key=os.getenv("JINAAI_API_KEY"),
        ),
    )


def test_chat_model():
    chat_model = llm_manager.resolve_chat_model(
        ChatModelConfig(provider=LLMProviders.OPENAI, model="gpt-4o")
    )

    res = chat_model.chat(
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
    embed_model = llm_manager.resolve_embedding_model(
        EmbeddingModelConfig(
            provider=LLMProviders.OPENAI,
            model="text-embedding-3-small",
            dimensions=1536,
        )
    )
    vector = embed_model.get_query_embedding("What is TiDB?")
    assert len(vector) == 1536


def test_reranker_model():
    reranker_model = llm_manager.resolve_reranker_model(
        RerankerModelConfig(
            provider=LLMProviders.JINA_AI, model="jina-reranker-v2-base-multilingual"
        )
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
