"""Bounded model HTTP clients without ambient proxies or automatic redirects."""

import httpx
import requests
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
    def __init__(self, endpoint):
        super().__init__()
        self.trust_env = False
        self.allowed_origin = endpoint_origin(endpoint)

    def request(self, method, url, **kwargs):
        if endpoint_origin(url) != self.allowed_origin:
            raise ProviderConfigurationError("Provider request changed destination")
        kwargs["allow_redirects"] = False
        kwargs["timeout"] = (10, 60)
        generation = current_generation.get()
        remaining = generation.before_outbound() if generation else None
        was_streaming = kwargs.get("stream", False)
        if remaining is not None:
            kwargs["timeout"] = (min(10, remaining), min(60, remaining))
            kwargs["stream"] = True
        response = super().request(method, url, **kwargs)
        if 300 <= response.status_code < 400:
            response.close()
            raise ProviderConfigurationError("Provider redirects are disabled")
        if remaining is not None:
            original = response.iter_content

            def bounded_content(chunk_size=1, decode_unicode=False):
                total = 0
                try:
                    for chunk in original(chunk_size, decode_unicode):
                        generation.check()
                        total += len(
                            chunk.encode() if isinstance(chunk, str) else chunk
                        )
                        if total > MAX_RESPONSE_BYTES:
                            raise GenerationExpired(
                                "Generation response budget exceeded"
                            )
                        yield chunk
                except BaseException:
                    response.close()
                    raise

            response.iter_content = bounded_content
            if not was_streaming:
                response.content
        return response
