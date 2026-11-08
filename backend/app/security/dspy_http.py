"""Bind DSPy's native OpenAI engine to the application's guarded HTTP path."""

from contextlib import asynccontextmanager, contextmanager

import httpx
from dspy.clients.engines import LM15Engine, AsyncLM15Engine
from dspy.lm15 import RouterConfig
from dspy._vendor.lm15.transports import TransportResponse, AsyncTransportResponse

from app.security.generation_http import (
    GenerationTransport,
    GenerationAsyncTransport,
)
from app.security.provider_config import endpoint_origin, ProviderConfigurationError


class ModelTransport:
    def __init__(self, endpoint, timeout):
        self.origin = endpoint_origin(endpoint)
        self.timeout = min(timeout, 60)

    def options(self, request):
        if endpoint_origin(request.url) != self.origin:
            raise ProviderConfigurationError("Provider request changed destination")
        return dict(
            method=request.method,
            url=request.url,
            headers=request.headers,
            content=request.body,
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


def openai_engines(endpoint, credential, timeout):
    # Explicit custom engines prevent automatic fallback to LiteLLM when
    # unrelated process-level API_BASE/proxy variables happen to be present.
    options = dict(
        env={},
        api_keys={"openai-chat": credential},
        base_urls={"openai-chat": endpoint},
    )
    return {
        "engine": LM15Engine(
            RouterConfig(**options, transport=ModelTransport(endpoint, timeout))
        ),
        "async_engine": AsyncLM15Engine(
            RouterConfig(**options, transport=AsyncModelTransport(endpoint, timeout))
        ),
    }
