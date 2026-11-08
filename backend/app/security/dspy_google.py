"""Use native Gemini codecs with the primary client's explicit Google identity."""

import asyncio
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx
from google.genai._transformers import t_model
from dspy.clients.engines import LM15Engine, AsyncLM15Engine
from dspy.lm15 import RouterConfig
from dspy._vendor.lm15.transports import AsyncTransportResponse

from app.security.dspy_http import ModelTransport, bind_native_capabilities
from app.security.generation_http import GenerationAsyncTransport
from app.security.google_http import bound_google_body, request_origin
from app.security.provider_config import ProviderConfigurationError


class GoogleModelTransport(ModelTransport):
    def __init__(self, api, model, max_tokens):
        super().__init__(api._http_options.base_url, api._http_options.timeout / 1000)
        self.api = api
        self.model_path = t_model(api, model)
        self.max_tokens = max_tokens

    def access_token(self):
        return self.api._access_token() if self.api.vertexai else None

    def options(self, request):
        return self.bound_options(request, self.access_token())

    def bound_options(self, request, access_token):
        if request_origin(request.url) != self.origin:
            raise ProviderConfigurationError("Provider request changed destination")
        if request.method != "POST":
            raise ProviderConfigurationError("Unexpected Google model operation")
        # Keep the exact primary SDK project, location, model path and API
        # version. The native codec supplies Gemini JSON and SSE decoding only.
        operation = urlsplit(request.url).path.rsplit(":", 1)[-1]
        if operation not in {"generateContent", "streamGenerateContent"}:
            raise ProviderConfigurationError("Unexpected Google model operation")
        target = self.api._build_request("post", f"{self.model_path}:{operation}", {})
        if request_origin(target.url) != self.origin:
            raise ProviderConfigurationError("Provider request changed destination")
        url = target.url + ("?alt=sse" if operation == "streamGenerateContent" else "")
        headers = [("Content-Type", "application/json")]
        if access_token:
            headers.append(("Authorization", f"Bearer {access_token}"))
            if self.api._credentials.quota_project_id:
                headers.append(
                    ("x-goog-user-project", self.api._credentials.quota_project_id)
                )
        else:
            headers.append(("x-goog-api-key", self.api.api_key))
        return dict(
            method="POST",
            url=url,
            headers=headers,
            content=bound_google_body(request.body, self.max_tokens),
            timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 10)),
        )


class AsyncGoogleModelTransport(GoogleModelTransport):
    @asynccontextmanager
    async def stream(self, request):
        # Refreshing a service account uses requests. Keep it off the event
        # loop and propagate the generation context into the worker thread.
        access_token = await asyncio.to_thread(self.access_token)
        options = self.bound_options(request, access_token)
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


def google_model_name(llm):
    # Provider choice is determined by the bound SDK client, not by whether a
    # valid model resource happens to include "models/" in its identifier.
    return f"gemini/{llm.model.rsplit('/', 1)[-1]}"


def google_engines(llm):
    api = llm._client._api_client
    # Credentials are attached exclusively by the transport. The placeholder
    # prevents the native codec from discovering keys or cloud credentials.
    options = dict(
        env={},
        api_keys={"gemini": "autoflow-explicit-google-transport"},
        base_urls={"gemini": api._http_options.base_url},
    )
    engines = {
        "engine": LM15Engine(
            RouterConfig(
                **options,
                transport=GoogleModelTransport(api, llm.model, llm._max_tokens),
            )
        ),
        "async_engine": AsyncLM15Engine(
            RouterConfig(
                **options,
                transport=AsyncGoogleModelTransport(api, llm.model, llm._max_tokens),
            )
        ),
    }
    return bind_native_capabilities(engines, google_model_name(llm))
