"""Apply the active request's deadline to model transports and streamed reads."""

import httpx
from app.security.generation import current_generation
from app.security.generation_limits import GenerationExpired

MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def prepare(request):
    scope = current_generation.get()
    if not scope or not scope.lease:
        return None
    remaining = scope.before_outbound()
    request.headers["Accept-Encoding"] = "identity"
    try:
        if len(request.content) > 2 * 1024 * 1024:
            raise GenerationExpired("Generation request body budget exceeded")
    except httpx.RequestNotRead:
        raise GenerationExpired("Streaming model uploads are disabled") from None
    previous = request.extensions.get("timeout", {})
    request.extensions["timeout"] = {
        key: min(previous.get(key) or 60, remaining, 60)
        for key in ("connect", "read", "write", "pool")
    }
    return scope


class LimitedStream(httpx.SyncByteStream):
    def __init__(self, stream, scope):
        self.stream, self.scope = stream, scope

    def __iter__(self):
        total = 0
        try:
            for chunk in self.stream:
                self.scope.check()
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise GenerationExpired("Generation response budget exceeded")
                yield chunk
        except BaseException:
            self.close()
            raise

    def close(self):
        self.stream.close()


class LimitedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, stream, scope):
        self.stream, self.scope = stream, scope

    async def __aiter__(self):
        total = 0
        try:
            async for chunk in self.stream:
                self.scope.check()
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise GenerationExpired("Generation response budget exceeded")
                yield chunk
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self):
        await self.stream.aclose()


class GenerationTransport(httpx.HTTPTransport):
    def handle_request(self, request):
        scope = prepare(request)
        response = super().handle_request(request)
        if scope:
            if (
                response.headers.get("content-encoding", "identity").lower()
                != "identity"
            ):
                response.close()
                raise GenerationExpired("Compressed model responses are disabled")
            response.stream = LimitedStream(response.stream, scope)
        return response


class GenerationAsyncTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request):
        scope = prepare(request)
        response = await super().handle_async_request(request)
        if scope:
            if (
                response.headers.get("content-encoding", "identity").lower()
                != "identity"
            ):
                await response.aclose()
                raise GenerationExpired("Compressed model responses are disabled")
            response.stream = LimitedAsyncStream(response.stream, scope)
        return response
