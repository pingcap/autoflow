import dspy
from app.security.provider_config import ProviderConfigurationError

from llama_index.core.base.llms.base import BaseLLM


def get_dspy_lm_by_llama_llm(llama_llm: BaseLLM) -> dspy.LM:
    """
    Get the dspy LM by the llama LLM.

    In this project, we use both llama-index and dspy, both of them have their own LLM implementation.
    This function can help us reduce the complexity of the code by converting the llama LLM to the dspy LLM.
    """
    match llama_llm.class_name():
        case "openai_llm":
            return _bounded_lm(
                model=f"openai/{llama_llm.model}",
                max_tokens=llama_llm.max_tokens,
                api_key=llama_llm.api_key,
                api_base=enforce_trailing_slash(llama_llm.api_base),
                num_retries=1,
            )
        case "OpenAILike":
            return _bounded_lm(
                model=f"openai/{llama_llm.model}",
                max_tokens=llama_llm.max_tokens,
                api_key=llama_llm.api_key,
                api_base=enforce_trailing_slash(llama_llm.api_base),
                model_type="chat" if llama_llm.is_chat_model else "text",
                num_retries=1,
            )
        case "GenAI":
            if "models/" in llama_llm.model:
                # For Gemini
                model_name = llama_llm.model.split("models/")[1]
                return _bounded_lm(
                    model=f"gemini/{model_name}",
                    max_tokens=llama_llm._max_tokens,
                    api_key=llama_llm._client._api_client.api_key,
                )
            else:
                # For Vertex AI
                return _bounded_lm(
                    model=f"vertex_ai/{llama_llm.model}",
                    max_tokens=llama_llm._max_tokens,
                    context_window=llama_llm.context_window,
                    temperature=llama_llm.temperature,
                    vertex_location=llama_llm._location,
                    vertex_credentials=llama_llm._credentials,
                )
        case "Bedrock_Converse_LLM":
            return _bounded_lm(
                model=f"bedrock/converse/{llama_llm.model}",
                # Notice: Bedrock's default max_tokens is 512, which is too small for the application.
                max_tokens=llama_llm.max_tokens or 8192,
                aws_access_key_id=llama_llm.aws_access_key_id,
                aws_secret_access_key=llama_llm.aws_secret_access_key,
                aws_region_name=llama_llm.region_name,
            )
        case "Ollama_llm":
            return _bounded_lm(
                # Keep Ollama's /api/chat protocol; DSPy's automatic native
                # engine treats api_base as an OpenAI-compatible /v1 base.
                engine="litellm",
                model=f"ollama_chat/{llama_llm.model}",
                api_base=llama_llm.base_url,
                api_key=_ollama_key(llama_llm),
                timeout=llama_llm.request_timeout,
                temperature=llama_llm.temperature,
                num_retries=1,
            )
        case "azure_openai_llm":
            return _bounded_lm(
                model=f"azure/{llama_llm.model}",
                max_tokens=llama_llm.max_tokens,
                temperature=llama_llm.temperature,
                api_key=llama_llm.api_key,
                api_base=enforce_trailing_slash(llama_llm.azure_endpoint),
                api_version=llama_llm.api_version,
                deployment_id=llama_llm.engine,
            )
        case _:
            raise ValueError(f"Got unknown LLM provider: {llama_llm.class_name()}")


def enforce_trailing_slash(url: str):
    if url.endswith("/"):
        return url
    return url + "/"


def _ollama_key(llama_llm):
    # resolve_llm supplies either the configured key or an explicit local-only
    # placeholder. Omitting it here lets LiteLLM inherit OLLAMA_API_KEY or a
    # process-wide key instead, even for an unauthenticated local endpoint.
    authorization = llama_llm.client._client.headers.get("Authorization", "")
    if not authorization.startswith("Bearer ") or not authorization[7:]:
        raise ProviderConfigurationError(
            "Ollama requires an explicit credential binding"
        )
    return authorization[7:]


def _bounded_lm(**kwargs) -> dspy.LM:
    kwargs.setdefault("timeout", 60)
    kwargs["num_retries"] = 1
    # User prompts must not be persisted in a shared on-disk DSPy cache.
    return dspy.LM(cache=False, **kwargs)
