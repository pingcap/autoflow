"""Request admission and lease lifetime, including disconnected SSE consumers."""

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import lru_cache
import ipaddress
from threading import Lock

import anyio
from fastapi import HTTPException
from redis import Redis
from starlette.responses import StreamingResponse, JSONResponse

from app.security.generation_limits import (
    GenerationLimiter,
    GenerationRateLimited,
    GenerationUnavailable,
    GenerationExpired,
)

current_generation = ContextVar("current_generation", default=None)


@lru_cache(maxsize=1)
def get_generation_limiter():
    from app.core.config import settings

    url = (
        settings.GENERATION_REDIS_URL.get_secret_value()
        if settings.GENERATION_REDIS_URL
        else settings.CELERY_BROKER_URL
    )
    return GenerationLimiter(
        Redis.from_url(
            url,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
            max_connections=40,
        ),
        settings.SECRET_KEY.get_secret_value(),
        settings.GENERATION_LIMITS,
        settings.GENERATION_QUOTA_NAMESPACE,
    )


def client_ip(request, trusted_cidrs):
    peer = request.client.host if request.client else "unknown"
    try:
        trusted = [ipaddress.ip_network(value) for value in trusted_cidrs]
        address = ipaddress.ip_address(peer)
        forwarded = request.headers.get("x-forwarded-for", "")
        if not any(address in network for network in trusted) or not forwarded:
            return str(address)
        if len(forwarded) > 1024:
            return str(address)
        chain = [ipaddress.ip_address(value.strip()) for value in forwarded.split(",")]
        if len(chain) > 16:
            return str(address)
        for candidate in reversed(chain):
            if not any(address in network for network in trusted):
                break
            address = candidate
        return str(address)
    except ValueError:
        return peer


class GenerationScope:
    def __init__(self, request, user, resource=None):
        self.request, self.user, self.resource = request, user, resource
        self.lease = None
        self.provider_calls = 0
        self.lock = Lock()

    def admit(self):
        from app.core.config import settings

        if self.lease is not None:
            return
        actor = (
            f"user:{self.user.id}"
            if self.user
            else f"guest:{self.request.state.browser_id}"
        )
        try:
            self.lease = get_generation_limiter().acquire(
                actor=actor,
                authenticated=self.user is not None,
                client_ip=client_ip(
                    self.request, settings.GENERATION_TRUSTED_PROXY_CIDRS
                ),
                resource=self.resource,
            )
        except GenerationRateLimited as error:
            raise HTTPException(
                429,
                "Generation limit reached. Try again later.",
                headers={"Retry-After": str(error.retry_after)},
            ) from None
        except GenerationUnavailable:
            raise HTTPException(
                503, "Generation temporarily unavailable.", headers={"Retry-After": "5"}
            ) from None

    def check(self):
        if self.lease:
            return self.lease.remaining()
        return None

    def before_outbound(self):
        remaining = self.check()
        if self.lease:
            with self.lock:
                self.provider_calls += 1
                if (
                    self.provider_calls
                    > self.lease.limiter.limits.provider_calls_per_request
                ):
                    raise GenerationExpired("Generation provider call budget exceeded")
        return remaining

    @contextmanager
    def activate(self):
        token = current_generation.set(self)
        try:
            self.check()
            yield
            self.check()
        finally:
            current_generation.reset(token)

    def close(self):
        if self.lease:
            self.lease.close()


class GenerationIterator:
    def __init__(self, iterator, scope):
        self.iterator, self.scope = iter(iterator), scope
        self.context = copy_context()
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        return self.context.run(self._next)

    def _next(self):
        # Starlette can use a different worker thread for each next() call.
        # Never keep a ContextVar token open across a generator yield.
        try:
            with self.scope.activate():
                return next(self.iterator)
        except GenerationExpired:
            self._close()
            from app.rag.chat.stream_protocol import ChatEvent
            from app.rag.types import ChatEventType

            return ChatEvent(
                event_type=ChatEventType.ERROR_PART,
                payload="The request timed out. Please try again.",
            )

    def close(self):
        if not self.closed:
            self.context.run(self._close)

    def _close(self):
        self.closed = True
        try:
            close = getattr(self.iterator, "close", None)
            if close:
                with self.scope.activate():
                    close()
        except GenerationExpired:
            close = getattr(self.iterator, "close", None)
            if close:
                close()
        finally:
            self.scope.close()


class GenerationStreamingResponse(StreamingResponse):
    def __init__(self, iterator, **kwargs):
        self.generation_iterator = iterator
        super().__init__(iterator, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # anyio's non-abandoning worker waits for a running next() before
            # reaching this point. Do not free its slot while it still runs.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(self.generation_iterator.close)


class GenerationBodyLimitMiddleware:
    def __init__(self, app, prefix="/api/v1", max_bytes=262144):
        self.app, self.prefix, self.max_bytes = app, prefix, max_bytes

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        limited = path == self.prefix + "/chats" or (
            path.startswith(self.prefix + "/chat-messages/")
            and path.endswith("/recommended-questions")
        )
        if scope["type"] != "http" or scope.get("method") != "POST" or not limited:
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        try:
            with anyio.fail_after(15):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > self.max_bytes:
                        return await JSONResponse(
                            {"detail": "Request body too large"}, 413
                        )(scope, receive, send)
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            return await JSONResponse({"detail": "Request body timed out"}, 408)(
                scope, receive, send
            )
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": b"".join(chunks),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, bounded_receive, send)
