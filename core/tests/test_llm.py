import logging
import os

from llama_index.core.schema import NodeWithScore, TextNode

from autoflow.models import (
    default_model_manager as model_manager,
    ProviderConfig,
    ModelProviders,
    LLMConfig,
    EmbeddingModelConfig,
    RerankerModelConfig,
)
from autoflow.chat import ChatMessage

logger = logging.getLogger(__name__)

model_manager.configure_provider(
    name=ModelProviders.OPENAI,
    config=ProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
)

model_manager.configure_provider(
    name=ModelProviders.JINA_AI,
    config=ProviderConfig(
        api_key=os.getenv("JINAAI_API_KEY"),
    ),
)


def test_llm():
    llm = model_manager.resolve_chat_model(
        LLMConfig(provider=ModelProviders.OPENAI, model="gpt-4o")
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
        EmbeddingModelConfig(
            provider=ModelProviders.OPENAI, model="text-embedding-3-small"
        )
    )
    vector = embed_model.get_query_embedding("What is TiDB?")
    assert len(vector) == 1536


def test_reranker_model():
    reranker_model = model_manager.resolve_reranker_model(
        RerankerModelConfig(
            provider=ModelProviders.JINA_AI, model="jina-reranker-v2-base-multilingual"
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
