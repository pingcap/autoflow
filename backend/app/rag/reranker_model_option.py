from typing import List
from pydantic import BaseModel

from app.types import RerankerProvider


class RerankerModelOption(BaseModel):
    provider: RerankerProvider
    provider_display_name: str | None = None
    provider_description: str | None = None
    provider_url: str | None = None
    default_reranker_model: str
    reranker_model_description: str
    default_top_n: int = 10
    default_credentials: str | dict = ""
    default_config: dict = {}
    config_description: str = ""
    credentials_display_name: str
    credentials_description: str
    credentials_type: str = "str"


admin_reranker_model_options: List[RerankerModelOption] = [
    RerankerModelOption(
        provider=RerankerProvider.JINA,
        provider_display_name="Jina AI",
        provider_description="We provide best-in-class embeddings, rerankers, LLM-reader and prompt optimizers, pioneering search AI for multimodal data.",
        provider_url="https://jina.ai",
        default_reranker_model="jina-reranker-v2-base-multilingual",
        reranker_model_description="Reference: https://jina.ai/reranker/",
        default_top_n=10,
        credentials_display_name="Jina API Key",
        credentials_description="You can get one from https://jina.ai/reranker/",
        credentials_type="str",
        default_credentials="jina_****",
    ),
    RerankerModelOption(
        provider=RerankerProvider.COHERE,
        provider_display_name="Cohere",
        provider_description="Cohere provides industry-leading large language models (LLMs) and RAG capabilities tailored to meet the needs of enterprise use cases that solve real-world problems.",
        provider_url="https://cohere.com/",
        default_reranker_model="rerank-multilingual-v3.0",
        reranker_model_description="Reference: https://docs.cohere.com/reference/rerank",
        default_top_n=10,
        credentials_display_name="Cohere API Key",
        credentials_description="You can get one from https://dashboard.cohere.com/api-keys",
        credentials_type="str",
        default_credentials="*****",
    ),
    RerankerModelOption(
        provider=RerankerProvider.BAISHENG,
        provider_display_name="BaiSheng",
        default_reranker_model="bge-reranker-v2-m3",
        reranker_model_description="",
        default_top_n=10,
        default_config={
            "api_url": "http://api.chat.prd.yumc.local/chat/v1/reranker",
        },
        credentials_display_name="BaiSheng API Key",
        credentials_description="",
        credentials_type="str",
        default_credentials="*****",
    ),
    RerankerModelOption(
        provider=RerankerProvider.LOCAL,
        provider_display_name="Local Reranker",
        provider_description="TIDB.AI's local reranker server, deployed on your own infrastructure and powered by sentence-transformers.",
        default_reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_model_description="Find more models in huggingface.",
        default_top_n=10,
        default_config={
            "api_url": "http://local-embedding-reranker:5001/api/v1/reranker",
        },
        config_description="api_url is the url of the tidb ai local reranker server.",
        credentials_display_name="Local Reranker API Key",
        credentials_description="Local Reranker server doesn't require an API key, set a dummy string here is ok.",
        credentials_type="str",
        default_credentials="dummy",
    ),
    RerankerModelOption(
        provider=RerankerProvider.VLLM,
        provider_display_name="vLLM",
        provider_description="vLLM is a fast and easy-to-use library for LLM inference and serving.",
        default_reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_model_description="",
        default_top_n=10,
        default_config={
            "api_url": "http://localhost:8000/v1/score",
        },
        config_description="api_url is the url of the vLLM server, ensure it can be accessed from this server",
        credentials_display_name="vLLM API Key",
        credentials_description="If you don't need an API key to access your rerank model, set a dummy string here is ok.",
        credentials_type="str",
        default_credentials="dummy",
    ),
    RerankerModelOption(
        provider=RerankerProvider.XINFERENCE,
        provider_display_name="Xinference Reranker",
        provider_description="Xorbits Inference (Xinference) is an open-source platform to streamline the operation and integration of a wide array of AI models.",
        default_reranker_model="bge-reranker-v2-m3",
        reranker_model_description="",
        default_top_n=10,
        default_config={
            "api_url": "http://localhost:9997/v1/rerank",
        },
        config_description="api_url is the url of the Xinference server, ensure it can be accessed from this server",
        credentials_display_name="Xinference API Key",
        credentials_description="If you don't need an API key to access your rerank model, set a dummy string here is ok.",
        credentials_type="str",
        default_credentials="dummy",
    ),
]
