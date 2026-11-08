"""Bounded model HTTP clients without ambient proxies or automatic redirects."""

import httpx
import requests
import time
from app.security.provider_config import endpoint_origin, ProviderConfigurationError
from app.security.generation import current_generation
from app.security.generation_limits import GenerationExpired
from app.security.generation_http import (
    GenerationTransport,
    GenerationAsyncTransport,
    MAX_RESPONSE_BYTES,
)


def openai_http_clients():
    options = {
        "trust_env": False,
        "follow_redirects": False,
        "timeout": httpx.Timeout(60, connect=10),
        "limits": httpx.Limits(max_connections=20, max_keepalive_connections=5),
    }
    return {
        "http_client": httpx.Client(
            transport=GenerationTransport(trust_env=False, limits=options["limits"]),
            **options,
        ),
        "async_http_client": httpx.AsyncClient(
            transport=GenerationAsyncTransport(
                trust_env=False, limits=options["limits"]
            ),
            **options,
        ),
    }


class ProviderSession(requests.Session):
    def __init__(
        self, endpoint, *, max_seconds=60, max_response_bytes=MAX_RESPONSE_BYTES
    ):
        super().__init__()
        self.trust_env = False
        self.allowed_origin = endpoint_origin(endpoint)
        self.max_seconds = max_seconds
        self.max_response_bytes = max_response_bytes

    def send(self, request, **kwargs):
        body = request.body
        if isinstance(body, str):
            body = body.encode()
        if body is not None and (
            not isinstance(body, bytes) or len(body) > 2 * 1024 * 1024
        ):
            raise GenerationExpired("Generation request body budget exceeded")
        return super().send(request, **kwargs)

    def request(self, method, url, **kwargs):
        if endpoint_origin(url) != self.allowed_origin:
            raise ProviderConfigurationError("Provider request changed destination")
        kwargs["allow_redirects"] = False
        generation = current_generation.get()
        remaining = generation.before_outbound() if generation else None
        was_streaming = kwargs.get("stream", False)
        seconds = (
            min(self.max_seconds, remaining)
            if remaining is not None
            else self.max_seconds
        )
        deadline = time.monotonic() + seconds
        kwargs["timeout"] = (min(10, seconds), min(60, seconds))
        kwargs["stream"] = True
        headers = requests.structures.CaseInsensitiveDict(kwargs.get("headers") or {})
        headers["Accept-Encoding"] = "identity"
        kwargs["headers"] = headers
        response = super().request(method, url, **kwargs)
        if 300 <= response.status_code < 400:
            response.close()
            raise ProviderConfigurationError("Provider redirects are disabled")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            response.close()
            raise GenerationExpired("Compressed model responses are disabled")
        original = response.iter_content

        def bounded_content(chunk_size=1, decode_unicode=False):
            total = 0
            try:
                for chunk in original(chunk_size, decode_unicode):
                    if generation:
                        generation.check()
                    if time.monotonic() > deadline:
                        raise GenerationExpired("Generation deadline exceeded")
                    total += len(chunk.encode() if isinstance(chunk, str) else chunk)
                    if total > self.max_response_bytes:
                        raise GenerationExpired("Generation response budget exceeded")
                    yield chunk
            except BaseException:
                response.close()
                raise

        response.iter_content = bounded_content
        if not was_streaming:
            response.content
        return response
