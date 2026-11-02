"""Bounded model HTTP clients without ambient proxies or automatic redirects."""

import httpx
import requests
from app.security.provider_config import endpoint_origin, ProviderConfigurationError


def openai_http_clients():
    options = {
        "trust_env": False,
        "follow_redirects": False,
        "timeout": httpx.Timeout(60, connect=10),
        "limits": httpx.Limits(max_connections=20, max_keepalive_connections=5),
    }
    return {
        "http_client": httpx.Client(**options),
        "async_http_client": httpx.AsyncClient(**options),
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
        response = super().request(method, url, **kwargs)
        if 300 <= response.status_code < 400:
            response.close()
            raise ProviderConfigurationError("Provider redirects are disabled")
        return response
