from typing import List
from pydantic import BaseModel

from app.types import EmbeddingProvider
from app.core.config import settings


class EmbeddingModelOption(BaseModel):
    provider: EmbeddingProvider
    provider_display_name: str | None = None
    provider_description: str | None = None
    provider_url: str | None = None
    default_embedding_model: str
    embedding_model_description: str
    default_config: dict = {}
    config_description: str = ""
    default_credentials: str | dict = ""
    credentials_display_name: str
    credentials_description: str
    credentials_type: str = "str"


admin_embed_model_options: List[EmbeddingModelOption] = [
    EmbeddingModelOption(
        provider=EmbeddingProvider.OPENAI,
        provider_display_name="OpenAI",
        provider_description="The OpenAI API provides a simple interface for developers to create an intelligence layer in their applications, powered by OpenAI's state of the art models.",
        provider_url="https://platform.openai.com",
        default_embedding_model="text-embedding-3-small",
        embedding_model_description=f"Find more information about OpenAI Embedding at https://platform.openai.com/docs/guides/embeddings",
        credentials_display_name="OpenAI API Key",
        credentials_description="The API key of OpenAI, you can find it in https://platform.openai.com/api-keys",
        credentials_type="str",
        default_credentials="sk-****",
    ),
    EmbeddingModelOption(
        provider=EmbeddingProvider.JINA,
        provider_display_name="JinaAI",
        provider_description="Jina AI provides multimodal, bilingual long-context embeddings for search and RAG",
        provider_url="https://jina.ai/embeddings/",
        default_embedding_model="jina-embeddings-v2-base-en",
        embedding_model_description=f"Find more information about Jina AI Embeddings at https://jina.ai/embeddings/",
        credentials_display_name="Jina API Key",
        credentials_description="The API key of Jina, you can find it in https://jina.ai/embeddings/",
        credentials_type="str",
        default_credentials="jina_****",
    ),
    EmbeddingModelOption(
        provider=EmbeddingProvider.COHERE,
        provider_display_name="Cohere",
        provider_description="Cohere provides industry-leading large language models (LLMs) and RAG capabilities tailored to meet the needs of enterprise use cases that solve real-world problems.",
        provider_url="https://cohere.com/embeddings",
        default_embedding_model="embed-multilingual-v3.0",
        embedding_model_description=f"Documentation: https://docs.cohere.com/docs/cohere-embed",
        credentials_display_name="Cohere API Key",
        credentials_description="You can get one from https://dashboard.cohere.com/api-keys",
        credentials_type="str",
        default_credentials="*****",
    ),
    EmbeddingModelOption(
        provider=EmbeddingProvider.OLLAMA,
        provider_display_name="Ollama",
        provider_description="Ollama is a lightweight framework for building and running large language models and embed models.",
        provider_url="https://ollama.com",
        default_embedding_model="nomic-embed-text",
        embedding_model_description=f"Documentation: https://ollama.com/blog/embedding-models",
        default_config={
            "api_base": "http://localhost:11434",
        },
        config_description="api_base is the base URL of the Ollama server, ensure it can be accessed from this server.",
        credentials_display_name="Ollama API Key",
        credentials_description="Ollama doesn't require an API key, set a dummy string here is ok",
        credentials_type="str",
        default_credentials="dummy",
    ),
    EmbeddingModelOption(
        provider=EmbeddingProvider.LOCAL,
        provider_display_name="Local Embedding",
        provider_description="Autoflow's local embedding server, deployed on your own infrastructure and powered by sentence-transformers.",
        default_embedding_model="BAAI/bge-m3",
        embedding_model_description="Find more models in huggingface.",
        default_config={
            "api_url": "http://local-embedding-reranker:5001/api/v1/embedding",
        },
        config_description="api_url is the embedding endpoint url serviced by the autoflow local embedding server.",
        credentials_display_name="Local Embedding API Key",
        credentials_description="Local Embedding server doesn't require an API key, set a dummy string here is ok.",
        credentials_type="str",
        default_credentials="dummy",
    ),
    EmbeddingModelOption(
        provider=EmbeddingProvider.OPENAI_LIKE,
        provider_display_name="OpenAI Like",
        provider_description="OpenAI-Like is a set of platforms that provide text embeddings similar to OpenAI. Such as ZhiPuAI.",
        provider_url="https://open.bigmodel.cn/dev/api/vector/embedding-3",
        default_embedding_model="embedding-3",
        embedding_model_description="",
        credentials_display_name="OpenAI Like API Key",
        credentials_description="The API key of OpenAI Like. For ZhipuAI, you can find it in https://open.bigmodel.cn/usercenter/apikeys",
        credentials_type="str",
        default_credentials="dummy",
    ),
]
