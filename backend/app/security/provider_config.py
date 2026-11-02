"""Operator-controlled model destinations and explicit credential binding."""

from copy import deepcopy
import json
import math
import os
import re
from urllib.parse import urlsplit

from fastapi import HTTPException
from app.core.config import settings


class ProviderConfigurationError(ValueError):
    """Safe public error, containing no submitted values."""


ENDPOINT_FIELDS = {"api_base", "base_url", "api_url", "azure_endpoint"}
DESTINATION_FIELDS = ENDPOINT_FIELDS | {
    "engine",
    "deployment_name",
    "api_version",
    "project",
    "location",
}
DEFAULTS = {
    "openai": ("api_base", "https://api.openai.com/v1"),
    "openai_like": ("api_base", "https://openrouter.ai/api/v1"),
    "giteeai": ("api_base", "https://ai.gitee.com/v1"),
    "ollama": ("base_url", "http://localhost:11434"),
    "jina": ("base_url", "https://api.jina.ai/v1"),
    "cohere": ("base_url", "https://api.cohere.com"),
    "vllm": ("base_url", "http://localhost:8000"),
    "xinference": ("base_url", "http://localhost:9997"),
}
PUBLIC_ORIGINS = {
    "openai": {"https://api.openai.com"},
    "openai_like": {
        "https://openrouter.ai",
        "https://api.openai.com",
        "https://open.bigmodel.cn",
    },
    "giteeai": {"https://ai.gitee.com"},
    "jina": {"https://api.jina.ai"},
    "cohere": {"https://api.cohere.com", "https://api.cohere.ai"},
}
COMMON = {
    "llm": {
        "temperature",
        "max_tokens",
        "context_window",
        "max_retries",
        "timeout",
        "system_prompt",
    },
    "embedding": {"embed_batch_size", "num_workers"},
    "reranker": set(),
}
PROVIDER_FIELDS = {
    "openai": {
        "api_base",
        "api_version",
        "dimensions",
        "mode",
        "additional_kwargs",
        "reasoning_effort",
        "strict",
    },
    "openai_like": {
        "api_base",
        "api_version",
        "is_chat_model",
        "is_function_calling_model",
        "additional_kwargs",
    },
    "giteeai": {
        "api_base",
        "is_chat_model",
        "is_function_calling_model",
        "additional_kwargs",
    },
    "azure_openai": {
        "azure_endpoint",
        "api_version",
        "engine",
        "deployment_name",
        "dimensions",
        "mode",
        "additional_kwargs",
    },
    "gemini": {"is_function_calling_model"},
    "vertex_ai": {"project", "location", "is_function_calling_model"},
    "anthropic_vertex": {"project", "location", "is_function_calling_model"},
    "bedrock": {"additional_kwargs", "normalize"},
    "ollama": {
        "base_url",
        "request_timeout",
        "json_mode",
        "additional_kwargs",
        "ollama_additional_kwargs",
        "is_function_calling_model",
        "thinking",
    },
    "jina": {"base_url", "task", "dimensions", "late_chunking", "embedding_type"},
    "cohere": {"base_url", "input_type", "embedding_type", "truncate", "truncation"},
    "local": {"api_url", "normalize_embeddings"},
    "baisheng": {"api_url"},
    "vllm": {"base_url"},
    "xinference": {"base_url"},
}
GENERATION_FIELDS = {
    "top_p",
    "top_k",
    "seed",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "num_predict",
    "repeat_penalty",
    "num_ctx",
    "max_completion_tokens",
    "max_output_tokens",
}


def endpoint_origin(value):
    if (
        not isinstance(value, str)
        or len(value) > 2048
        or re.search(r"[\x00-\x20\\]", value)
    ):
        raise ProviderConfigurationError("Invalid provider endpoint")
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError()
        host = parsed.hostname.encode("idna").decode("ascii").lower()
        if not re.fullmatch(r"[a-z0-9.-]+|[a-f0-9:]+", host) or host.endswith("."):
            raise ValueError()
        port = parsed.port
        if ":" in host:
            host = f"[{host}]"
        suffix = (
            ""
            if port is None or port == (443 if parsed.scheme == "https" else 80)
            else f":{port}"
        )
        return f"{parsed.scheme}://{host}{suffix}"
    except (ValueError, UnicodeError):
        raise ProviderConfigurationError("Invalid provider endpoint") from None


def _matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _bounded_number(value, low, high):
    return (
        type(value) in (int, float)
        and (type(value) is int or math.isfinite(value))
        and low <= value <= high
    )


def prepare_provider_config(kind, provider, model, config, credentials):
    provider = str(getattr(provider, "value", provider))
    if provider not in PROVIDER_FIELDS or kind not in COMMON:
        raise ProviderConfigurationError("Unsupported model provider")
    if (
        not isinstance(model, str)
        or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,256}", model)
        or model.startswith("/")
        or "://" in model
        or ".." in model.split("/")
    ):
        raise ProviderConfigurationError("Invalid model identifier")
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ProviderConfigurationError("Model configuration must be an object")
    try:
        if len(json.dumps(config, allow_nan=False).encode()) > 64 * 1024:
            raise ValueError()
    except (ValueError, TypeError, RecursionError):
        raise ProviderConfigurationError(
            "Model configuration is invalid or too large"
        ) from None
    result = deepcopy(config)
    if (
        provider == "gemini"
        and os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false") != "false"
    ):
        raise ProviderConfigurationError(
            "Gemini cannot inherit ambient Vertex credentials; configure an explicit Vertex provider"
        )
    if provider == "ollama" and kind == "embedding" and "api_base" in result:
        if "base_url" in result:
            raise ProviderConfigurationError("Specify only one Ollama endpoint")
        result["base_url"] = result.pop("api_base")
    if set(result) - (COMMON[kind] | PROVIDER_FIELDS[provider]):
        raise ProviderConfigurationError("Unsupported model configuration fields")
    for field in ("additional_kwargs", "ollama_additional_kwargs"):
        if field in result:
            value = result[field]
            if not isinstance(value, dict) or set(value) - GENERATION_FIELDS:
                raise ProviderConfigurationError("Unsupported generation options")
            for token_limit in (
                "num_predict",
                "max_completion_tokens",
                "max_output_tokens",
            ):
                if token_limit in value and not _bounded_number(
                    value[token_limit], 1, 8192
                ):
                    raise ProviderConfigurationError(
                        "Output token limit must be between 1 and 8192"
                    )
            if "num_ctx" in value and not _bounded_number(value["num_ctx"], 1, 131072):
                raise ProviderConfigurationError("Invalid model context limit")
    for field, low, high in [
        ("max_tokens", 1, 8192),
        ("context_window", 1, 1048576),
        ("max_retries", 0, 2),
        ("timeout", 1, 120),
        ("request_timeout", 1, 120),
        ("embed_batch_size", 1, 128),
        ("num_workers", 1, 8),
        ("temperature", 0, 2),
    ]:
        if field in result and not _bounded_number(result[field], low, high):
            raise ProviderConfigurationError("Invalid model resource limit")
    if provider in DEFAULTS:
        field, default = DEFAULTS[provider]
        if kind == "embedding" and provider == "openai_like":
            default = "https://open.bigmodel.cn/api/paas/v4"
        result.setdefault(field, default)
    if provider == "local":
        result.setdefault(
            "api_url",
            f"http://local-embedding-reranker:5001/api/v1/{'embedding' if kind == 'embedding' else 'reranker'}",
        )
    if provider in {"azure_openai", "baisheng"} and not result.get(
        "azure_endpoint" if provider == "azure_openai" else "api_url"
    ):
        raise ProviderConfigurationError("An explicit provider endpoint is required")
    allowed = PUBLIC_ORIGINS.get(provider, set()) | {
        endpoint_origin(origin) for origin in settings.PROVIDER_ALLOWED_ORIGINS
    }
    for field in ENDPOINT_FIELDS & result.keys():
        if endpoint_origin(result[field]) not in allowed:
            raise ProviderConfigurationError(
                "Provider endpoint is not registered in PROVIDER_ALLOWED_ORIGINS"
            )
    if provider in {"vertex_ai", "anthropic_vertex"}:
        if (
            not isinstance(credentials, dict)
            or credentials.get("type") != "service_account"
            or credentials.get("token_uri") != "https://oauth2.googleapis.com/token"
            or credentials.get("universe_domain", "googleapis.com") != "googleapis.com"
        ):
            raise ProviderConfigurationError(
                "Explicit Google service-account credentials with the standard token endpoint are required"
            )
        if not _matches(
            r"[a-z][a-z0-9-]{0,62}", result.get("location", "us-west1")
        ) or not _matches(
            r"[a-z][a-z0-9-]{4,61}[a-z0-9]",
            credentials.get("project_id") or result.get("project") or "",
        ):
            raise ProviderConfigurationError("Invalid Vertex project or location")
    elif provider == "bedrock":
        if (
            not isinstance(credentials, dict)
            or any(
                not isinstance(credentials.get(k), str) or not credentials[k]
                for k in (
                    "aws_access_key_id",
                    "aws_secret_access_key",
                    "aws_region_name",
                )
            )
            or not re.fullmatch(
                r"[a-z]{2}(?:-[a-z]+)+-\d", credentials["aws_region_name"]
            )
        ):
            raise ProviderConfigurationError(
                "Explicit AWS credentials and region are required"
            )
    elif provider == "local":
        if not isinstance(credentials, str) or not re.fullmatch(
            r"[A-Za-z0-9._~+/-]{32,128}={0,2}", credentials
        ):
            raise ProviderConfigurationError(
                "The local model service requires its explicit SIDECAR_AUTH_TOKEN"
            )
    elif provider not in {"ollama", "vllm", "xinference"}:
        if not isinstance(credentials, str) or not credentials.strip():
            raise ProviderConfigurationError(
                "Explicit provider credentials are required; server credentials are not inherited"
            )
    if kind == "llm" and provider != "ollama":
        result.setdefault("max_tokens", 4096)
        result.setdefault("max_retries", 1)
    elif kind == "llm":
        result.setdefault("additional_kwargs", {}).setdefault("num_predict", 4096)
        result.setdefault("request_timeout", 60)
    return result


def validate_provider_write(kind, item, changes=None):
    """Validate before mutations; require fresh credentials for destination changes."""
    try:
        new_config = (changes or {}).get("config", item.config)
        if changes is not None and "config" in changes:
            old, new = item.config or {}, new_config or {}
            if not isinstance(old, dict) or not isinstance(new, dict):
                raise ProviderConfigurationError(
                    "Model configuration must be an object"
                )
            if any(
                old.get(k) != new.get(k) for k in DESTINATION_FIELDS
            ) and not changes.get("credentials"):
                raise ProviderConfigurationError(
                    "Changing the provider destination requires fresh credentials"
                )
        return prepare_provider_config(
            kind,
            item.provider,
            item.model,
            new_config,
            (changes or {}).get("credentials", item.credentials),
        )
    except ProviderConfigurationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
