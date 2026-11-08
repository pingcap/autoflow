"""Request-scoped Langfuse clients, without the SDK's process-wide singleton.

The routing handlers stay registered once; context variables select the current
chat's handlers. Closing one chat never removes another chat's instrumentation.
"""

from contextvars import ContextVar
import logging
from threading import Lock

from langfuse import Langfuse
from langfuse.llama_index import LlamaIndexInstrumentor
from langfuse.llama_index._context import (
    InstrumentorContext,
    default_context,
    langfuse_instrumentor_context,
)
from langfuse.llama_index._event_handler import LlamaIndexEventHandler
from langfuse.llama_index._span_handler import LlamaIndexSpanHandler
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler

from app.core.config import settings
from app.security.langfuse_http import langfuse_http_client
from app.security.langfuse_settings import DEFAULT_HOST, RuntimeConfiguration
from app.security.trace_lifecycle import TraceLifecycle

_active = ContextVar("autoflow_langfuse_instrumentor", default=None)
_registration_lock = Lock()
_client_init_lock = Lock()


class RoutingEvents(BaseEventHandler):
    def handle(self, event, **kwargs):
        current = _active.get()
        if current is not None:
            current._event_handler.handle(event, **kwargs)


class RoutingSpans(BaseSpanHandler):
    def span_enter(self, *args, **kwargs):
        current = _active.get()
        if current is not None:
            return current._span_handler.span_enter(*args, **kwargs)

    def span_exit(self, *args, **kwargs):
        current = _active.get()
        if current is not None:
            return current._span_handler.span_exit(*args, **kwargs)

    def span_drop(self, *args, **kwargs):
        current = _active.get()
        if current is not None:
            return current._span_handler.span_drop(*args, **kwargs)

    def new_span(self, *args, **kwargs):
        return None

    def prepare_to_exit_span(self, *args, **kwargs):
        return None

    def prepare_to_drop_span(self, *args, **kwargs):
        return None


class BoundLangfuseInstrumentor(LlamaIndexInstrumentor):
    def __init__(self, config: RuntimeConfiguration):
        self.config = config
        self._context = InstrumentorContext()
        self._tokens = None
        self._langfuse = None
        self._http = None
        self._sdk_httpx_handlers = []
        self._lifecycle = None

    def start(self):
        if self._tokens is not None:
            return
        host = self.config.host if self.config.enabled else DEFAULT_HOST
        self._http = langfuse_http_client(host)
        try:
            # This SDK adds a logger handler during construction; track only ours.
            with _client_init_lock:
                httpx_logger = logging.getLogger("httpx")
                previous_handlers = set(httpx_logger.handlers)
                previous_level = httpx_logger.level
                try:
                    self._langfuse = Langfuse(
                        host=host,
                        public_key=self.config.public_key or "disabled",
                        secret_key=self.config.secret_key or "disabled",
                        enabled=self.config.enabled,
                        httpx_client=self._http,
                        threads=1,
                        max_retries=1,
                        timeout=10,
                        flush_interval=0.2,
                        environment=settings.ENVIRONMENT.value,
                    )
                finally:
                    self._sdk_httpx_handlers = [
                        h for h in httpx_logger.handlers if h not in previous_handlers
                    ]
                    httpx_logger.setLevel(previous_level)
            self._lifecycle = TraceLifecycle(self._langfuse, self._http._transport)
            self._span_handler = LlamaIndexSpanHandler(langfuse_client=self._langfuse)
            self._event_handler = LlamaIndexEventHandler(langfuse_client=self._langfuse)
            with _registration_lock:
                dispatcher = get_dispatcher()
                if not any(
                    isinstance(h, RoutingEvents) for h in dispatcher.event_handlers
                ):
                    dispatcher.add_event_handler(RoutingEvents())
                if not any(
                    isinstance(h, RoutingSpans) for h in dispatcher.span_handlers
                ):
                    dispatcher.add_span_handler(RoutingSpans())
            self._tokens = (
                _active.set(self),
                langfuse_instrumentor_context.set({**default_context}),
            )
        except BaseException:
            self._close_client()
            raise

    @property
    def _is_instrumented(self):
        return _active.get() is self

    def stop(self):
        if self._tokens is None:
            return
        active_token, context_token = self._tokens
        self._tokens = None
        try:
            _active.reset(active_token)
            langfuse_instrumentor_context.reset(context_token)
        finally:
            self._close_client()

    def _close_client(self):
        try:
            if self._lifecycle is not None:
                # Langfuse.shutdown() removes every httpx log handler, including
                # unrelated application handlers. Shut down only owned resources.
                self._lifecycle.close()
        finally:
            for handler in self._sdk_httpx_handlers:
                logging.getLogger("httpx").removeHandler(handler)
                handler.close()
            self._sdk_httpx_handlers = []
            if self._http is not None:
                self._http.close()

    def flush(self):
        if self._lifecycle is not None:
            self._lifecycle.flush()
