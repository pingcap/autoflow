"""Bind DSPy's native OpenAI engine to the application's guarded HTTP path."""

from contextlib import asynccontextmanager, contextmanager
import json

import httpx
import dspy
from dspy.clients.capabilities import capabilities
from dspy.clients.engines import LM15Engine, AsyncLM15Engine
from dspy.lm15 import RouterConfig
from dspy._vendor.lm15.transports import TransportResponse, AsyncTransportResponse

from app.security.generation_http import (
    GenerationTransport,
    GenerationAsyncTransport,
)
from app.security.provider_config import endpoint_origin, ProviderConfigurationError
from app.security.generation_limits import GenerationExpired


def bind_output_limit(payload, token_parameter, limit):
    for field in ("max_tokens", "max_completion_tokens"):
        if field in payload and field != token_parameter:
            payload[token_parameter] = payload.pop(field)
    requested = payload.get(token_parameter, limit)
    if type(requested) is not int or requested < 1:
        raise ProviderConfigurationError("Invalid model output limit")
    payload[token_parameter] = min(requested, limit)


class ModelTransport:
    def __init__(
        self, endpoint, timeout, *, output_limit=None, token_parameter="max_tokens"
    ):
        self.origin = endpoint_origin(endpoint)
        self.timeout = min(timeout, 60)
        self.output_limit = output_limit
        self.token_parameter = token_parameter

    def options(self, request):
        if endpoint_origin(request.url) != self.origin:
            raise ProviderConfigurationError("Provider request changed destination")
        content = request.body
        headers = request.headers
        if self.output_limit is not None:
            if request.method != "POST":
                raise ProviderConfigurationError("Unexpected model operation")
            if len(content) > 2 * 1024 * 1024:
                raise GenerationExpired("Generation request body budget exceeded")
            payload = json.loads(content)
            bind_output_limit(payload, self.token_parameter, self.output_limit)
            content = json.dumps(payload).encode()
            headers = [
                (key, value)
                for key, value in headers
                if key.lower() not in {"content-length", "content-encoding"}
            ]
        return dict(
            method=request.method,
            url=request.url,
            headers=headers,
            content=content,
            timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 10)),
        )

    @contextmanager
    def stream(self, request):
        options = self.options(request)
        # These transports are borrowed by DSPy's engines. Per-request context
        # ownership closes sockets on errors and early stream exit, including
        # when callers copy an LM or use it on multiple event loops.
        with httpx.Client(
            transport=GenerationTransport(trust_env=False),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            with client.stream(**options) as response:
                if 300 <= response.status_code < 400:
                    raise ProviderConfigurationError("Provider redirects are disabled")
                yield TransportResponse(
                    status=response.status_code,
                    reason=response.reason_phrase,
                    headers=list(response.headers.multi_items()),
                    http_version=response.http_version,
                    chunks=response.iter_raw(),
                    release=lambda _consumed: response.close(),
                )


class AsyncModelTransport(ModelTransport):
    @asynccontextmanager
    async def stream(self, request):
        options = self.options(request)
        async with httpx.AsyncClient(
            transport=GenerationAsyncTransport(trust_env=False),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            async with client.stream(**options) as response:
                if 300 <= response.status_code < 400:
                    raise ProviderConfigurationError("Provider redirects are disabled")

                async def release(_consumed):
                    await response.aclose()

                yield AsyncTransportResponse(
                    status=response.status_code,
                    reason=response.reason_phrase,
                    headers=list(response.headers.multi_items()),
                    http_version=response.http_version,
                    chunks=response.aiter_raw(),
                    release=release,
                )


def openai_engines(
    endpoint, credential, timeout, *, model, max_tokens, token_parameter
):
    # Explicit custom engines prevent automatic fallback to LiteLLM when
    # unrelated process-level API_BASE/proxy variables happen to be present.
    options = dict(
        env={},
        api_keys={"openai-chat": credential},
        base_urls={"openai-chat": endpoint},
    )
    engines = {
        "engine": LM15Engine(
            RouterConfig(
                **options,
                transport=ModelTransport(
                    endpoint,
                    timeout,
                    output_limit=max_tokens,
                    token_parameter=token_parameter,
                ),
            )
        ),
        "async_engine": AsyncLM15Engine(
            RouterConfig(
                **options,
                transport=AsyncModelTransport(
                    endpoint,
                    timeout,
                    output_limit=max_tokens,
                    token_parameter=token_parameter,
                ),
            )
        ),
    }
    return bind_native_capabilities(engines, f"openai/{model}")


def bind_native_capabilities(engines, model):
    # Custom engine objects do not inherit DSPy's adapter capability hints.
    # Resolve the native hints offline (app bootstrap pins the bundled metadata
    # snapshot); do not send a discovery request or assume every model supports
    # schemas/tools. This LM is used for planning only, never for inference.
    planner = dspy.LM(
        model=model,
        engine="lm15",
        cache=False,
        temperature=None,
        max_tokens=None,
    )
    try:
        hints = capabilities(planner)
    finally:
        planner.close()
    for engine in engines.values():
        engine.supports_function_calling = hints.function_calling
        engine.supports_reasoning = hints.reasoning
        engine.supports_response_schema = hints.response_schema
        engine.supported_params = hints.params
    return engines
