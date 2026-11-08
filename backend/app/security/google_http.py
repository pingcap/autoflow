"""Bind Google SDK requests and credential renewal to owned, bounded transports."""

import json
from urllib.parse import urlsplit

import httpx
from google.auth.transport.requests import Request
from google.genai.client import DebugConfig
from google.genai.types import HttpOptions, HttpRetryOptions
from google.oauth2 import service_account

from app.security.generation_http import GenerationTransport, GenerationAsyncTransport
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import endpoint_origin, ProviderConfigurationError
from app.security.provider_http import ProviderSession

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"


def vertex_endpoint(location):
    host = "aiplatform" if location == "global" else f"{location}-aiplatform"
    return f"https://{host}.googleapis.com"


def request_origin(url):
    # Endpoint configuration deliberately forbids queries. Actual Google SSE
    # requests use ?alt=sse, so remove only the query for this origin comparison.
    parsed = urlsplit(str(url))
    return endpoint_origin(parsed._replace(query="").geturl())


def bound_google_body(content, max_tokens):
    if len(content) > 2 * 1024 * 1024:
        raise GenerationExpired("Generation request body budget exceeded")
    payload = json.loads(content)
    config = payload.setdefault("generationConfig", {})
    requested = config.get("maxOutputTokens", max_tokens)
    if type(requested) is not int or requested < 1:
        raise ProviderConfigurationError("Invalid model output limit")
    config["maxOutputTokens"] = min(requested, max_tokens)
    return json.dumps(payload).encode()


class _GoogleRequest:
    def __init__(self, endpoint, max_tokens):
        self.origin = endpoint_origin(endpoint)
        self.max_tokens = max_tokens
        super().__init__(trust_env=False)

    def bind(self, request):
        if request_origin(request.url) != self.origin:
            raise ProviderConfigurationError("Provider request changed destination")
        if request.url.path.endswith((":generateContent", ":streamGenerateContent")):
            try:
                body = bound_google_body(request.content, self.max_tokens)
            except httpx.RequestNotRead:
                raise GenerationExpired(
                    "Streaming model uploads are disabled"
                ) from None
            headers = request.headers.copy()
            for name in ("content-length", "content-encoding"):
                headers.pop(name, None)
            request = httpx.Request(
                request.method,
                request.url,
                headers=headers,
                content=body,
                extensions=request.extensions,
            )
        return request


class GoogleTransport(_GoogleRequest, GenerationTransport):
    def handle_request(self, request):
        response = super().handle_request(self.bind(request))
        if 300 <= response.status_code < 400:
            response.close()
            raise ProviderConfigurationError("Provider redirects are disabled")
        return response


class AsyncGoogleTransport(_GoogleRequest, GenerationAsyncTransport):
    async def handle_async_request(self, request):
        response = await super().handle_async_request(self.bind(request))
        if 300 <= response.status_code < 400:
            await response.aclose()
            raise ProviderConfigurationError("Provider redirects are disabled")
        return response


def google_options(endpoint, *, vertex, timeout, max_tokens):
    return dict(
        # Disable ambient recording/replay of prompts and responses.
        debug_config=DebugConfig(
            client_mode=None, replays_directory=None, replay_id=None
        ),
        http_options=HttpOptions(
            base_url=endpoint,
            api_version="v1beta1" if vertex else "v1beta",
            timeout=int(min(timeout, 60) * 1000),
            # LlamaIndex owns the configured retry policy. Avoid multiplying it
            # by the Google SDK's independent default retry loop.
            retry_options=HttpRetryOptions(attempts=1),
            client_args={
                "trust_env": False,
                "follow_redirects": False,
                "transport": GoogleTransport(endpoint, max_tokens),
            },
            async_client_args={
                "trust_env": False,
                "follow_redirects": False,
                # Google's supported custom transport path selects HTTPX in
                # async calls too, instead of its separate aiohttp session.
                "transport": AsyncGoogleTransport(endpoint, max_tokens),
            },
        ),
    )


class BoundServiceAccount(service_account.Credentials):
    def refresh(self, request=None):
        # Google creates a new default requests session on later token expiry.
        # Always replace that request object, including on copied credentials.
        if self._token_uri != TOKEN_ENDPOINT:
            raise ProviderConfigurationError("Unexpected Google token endpoint")
        with ProviderSession(TOKEN_ENDPOINT) as session:
            super().refresh(Request(session=session))
