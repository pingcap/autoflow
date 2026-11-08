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
            return _openai_chat_lm(llama_llm)
        case "OpenAILike":
            if llama_llm.is_chat_model:
                return _openai_chat_lm(llama_llm)
            return _bounded_lm(
                model=f"openai/{llama_llm.model}",
                max_tokens=llama_llm.max_tokens,
                api_key=llama_llm.api_key,
                api_base=enforce_trailing_slash(llama_llm.api_base),
                model_type="chat" if llama_llm.is_chat_model else "text",
                num_retries=1,
            )
        case "GenAI":
            from app.security.dspy_google import google_engines, google_model_name

            return dspy.LM(
                model=google_model_name(llama_llm),
                max_tokens=None,
                temperature=llama_llm.temperature,
                cache=False,
                num_retries=1,
                **google_engines(llama_llm),
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
            from app.security.dspy_ollama import ollama_engines

            return dspy.LM(
                model=f"ollama_chat/{llama_llm.model}",
                cache=False,
                temperature=llama_llm.temperature,
                max_tokens=llama_llm.additional_kwargs.get("num_predict", 4096),
                num_retries=1,
                **ollama_engines(
                    llama_llm.model,
                    llama_llm.base_url,
                    _ollama_key(llama_llm),
                    llama_llm.request_timeout,
                    function_calling=llama_llm.is_function_calling_model,
                    thinking=llama_llm.thinking,
                    generation_options={
                        "num_ctx": llama_llm.context_window,
                        **llama_llm.additional_kwargs,
                    },
                    json_mode=llama_llm.json_mode,
                ),
            )
        case "azure_openai_llm":
            from app.security.dspy_azure import azure_engines

            return dspy.LM(
                model=f"azure/{llama_llm.model}",
                # DSPy recommends >=16k for reasoning models at construction.
                # Enforce the configured smaller allowance on the wire instead.
                max_tokens=None,
                temperature=llama_llm.temperature,
                cache=False,
                num_retries=1,
                **azure_engines(
                    llama_llm.azure_endpoint,
                    llama_llm.api_key,
                    llama_llm.timeout,
                    model=llama_llm.model,
                    deployment=llama_llm.engine,
                    api_version=llama_llm.api_version,
                    max_tokens=llama_llm.max_tokens or 4096,
                    token_parameter=(
                        "max_completion_tokens"
                        if "max_completion_tokens" in llama_llm._get_model_kwargs()
                        else "max_tokens"
                    ),
                ),
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


def _openai_chat_lm(llama_llm):
    from app.security.dspy_http import openai_engines

    return dspy.LM(
        model=f"openai/{llama_llm.model}",
        # The guarded transport enforces the configured output cap, including
        # reasoning models whose recommended SDK defaults exceed that cap.
        max_tokens=None,
        cache=False,
        num_retries=1,
        **openai_engines(
            enforce_trailing_slash(llama_llm.api_base),
            llama_llm.api_key,
            llama_llm.timeout,
            model=llama_llm.model,
            max_tokens=llama_llm.max_tokens or 4096,
            token_parameter=(
                "max_completion_tokens"
                if "max_completion_tokens" in llama_llm._get_model_kwargs()
                else "max_tokens"
            ),
        ),
    )
