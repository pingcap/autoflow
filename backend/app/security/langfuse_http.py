"""Destination-bound Langfuse HTTP without ambient proxies or redirects."""

import time
import httpx

from app.security.langfuse_settings import LangfuseConfigurationError
from app.security.provider_config import endpoint_origin


class LangfuseTransport(httpx.HTTPTransport):
    def __init__(self, host):
        super().__init__(trust_env=False, retries=0)
        self.origin = endpoint_origin(host)

    def handle_request(self, request):
        if endpoint_origin(str(request.url)) != self.origin:
            raise LangfuseConfigurationError("Langfuse request changed destination")
        request.headers["Accept-Encoding"] = "identity"
        deadline = time.monotonic() + 30
        response = super().handle_request(request)
        try:
            if 300 <= response.status_code < 400:
                raise LangfuseConfigurationError("Langfuse redirects are disabled")
            if response.status_code >= 400:
                # The SDK logs API error bodies. Do not let a reflected credential
                # or submitted trace content be copied into diagnostic logs.
                return httpx.Response(
                    response.status_code, json={"message": "Langfuse request failed"}
                )
            if (
                response.headers.get("content-encoding", "identity").lower()
                != "identity"
            ):
                raise LangfuseConfigurationError(
                    "Compressed Langfuse responses are disabled"
                )
            chunks, size = [], 0
            for chunk in response.stream:
                size += len(chunk)
                if size > 65536 or time.monotonic() > deadline:
                    raise LangfuseConfigurationError("Langfuse response limit exceeded")
                chunks.append(chunk)
            return httpx.Response(
                response.status_code, headers=response.headers, content=b"".join(chunks)
            )
        finally:
            response.close()


def langfuse_http_client(host):
    return httpx.Client(
        transport=LangfuseTransport(host),
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(10, connect=5),
        limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
    )
