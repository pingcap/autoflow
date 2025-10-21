import dspy

from llama_index.core.base.llms.base import BaseLLM


def get_dspy_lm_by_llama_llm(llama_llm: BaseLLM) -> dspy.LM:
    """
    Get the dspy LM by the llama LLM.

    In this project, we use both llama-index and dspy, both of them have their own LLM implementation.
    This function can help us reduce the complexity of the code by converting the llama LLM to the dspy LLM.
    """
    match llama_llm.class_name():
        case "openai_llm":
            return dspy.LM(
                model=f"openai/{llama_llm.model}",
                max_tokens=llama_llm.max_tokens,
                api_key=llama_llm.api_key,
                api_base=enforce_trailing_slash(llama_llm.api_base),
                num_retries=3,
            )
        case "OpenAILike":
            return dspy.LM(
                model=f"openai/{llama_llm.model}",
                max_tokens=llama_llm.max_tokens,
                api_key=llama_llm.api_key,
                api_base=enforce_trailing_slash(llama_llm.api_base),
                model_type="chat" if llama_llm.is_chat_model else "text",
                num_retries=3,
            )
        case "GenAI":
            if "models/" in llama_llm.model:
                # For Gemini
                model_name = llama_llm.model.split("models/")[1]
                return dspy.LM(
                    model=f"gemini/{model_name}",
                    max_tokens=llama_llm._max_tokens,
                    api_key=llama_llm._client._api_client.api_key,
                )
            else:
                # For Vertex AI
                return dspy.LM(
                    model=f"vertex_ai/{llama_llm.model}",
                    max_tokens=llama_llm._max_tokens,
                    context_window=llama_llm.context_window,
                    temperature=llama_llm.temperature,
                    vertex_location=llama_llm._location,
                    vertex_credentials=llama_llm._credentials,
                )
        case "Bedrock_Converse_LLM":
            return dspy.LM(
                model=f"bedrock/converse/{llama_llm.model}",
                # Notice: Bedrock's default max_tokens is 512, which is too small for the application.
                max_tokens=llama_llm.max_tokens or 8192,
                aws_access_key_id=llama_llm.aws_access_key_id,
                aws_secret_access_key=llama_llm.aws_secret_access_key,
                aws_region_name=llama_llm.region_name,
            )
        case "Ollama_llm":
            return dspy.LM(
                model=f"ollama_chat/{llama_llm.model}",
                api_base=llama_llm.base_url,
                timeout=llama_llm.request_timeout,
                temperature=llama_llm.temperature,
                num_retries=3,
            )
        case "azure_openai_llm":
            return dspy.LM(
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
