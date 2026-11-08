"""Cancelable, deadline-aware sockets owned by one tracing client.

Use httpcore's network interface so slow headers and partial writes cannot reset
the response deadline. DNS resolution still uses the operating system resolver.
"""

from contextvars import ContextVar
from threading import Lock
import socket
import time

import httpcore

from app.security.langfuse_settings import LangfuseConfigurationError


class TraceNetworkBackend(httpcore.NetworkBackend):
    def __init__(self):
        self.delegate = httpcore.SyncBackend()
        self.deadline = ContextVar("trace_http_deadline", default=None)
        self.lock = Lock()
        self.streams = set()
        self.cancelled = False

    def timeout(self, value):
        if self.cancelled:
            raise LangfuseConfigurationError("Tracing connection cancelled")
        deadline = self.deadline.get()
        remaining = deadline - time.monotonic() if deadline is not None else 10
        if remaining <= 0:
            raise LangfuseConfigurationError("Tracing request deadline exceeded")
        return min(value if value is not None else 10, remaining, 10)

    def wrap(self, stream):
        result = TraceNetworkStream(stream, self)
        with self.lock:
            cancelled = self.cancelled
            if not cancelled:
                self.streams.add(result)
        if cancelled:
            stream.close()
            raise LangfuseConfigurationError("Tracing connection cancelled")
        return result

    def connect_tcp(
        self, host, port, timeout=None, local_address=None, socket_options=None
    ):
        return self.wrap(
            self.delegate.connect_tcp(
                host, port, self.timeout(timeout), local_address, socket_options
            )
        )

    def connect_unix_socket(self, *args, **kwargs):
        raise LangfuseConfigurationError("Tracing Unix sockets are disabled")

    def abort(self):
        with self.lock:
            self.cancelled = True
            streams = list(self.streams)
        for stream in streams:
            stream.abort()


class TraceNetworkStream(httpcore.NetworkStream):
    def __init__(self, stream, backend):
        self.stream, self.backend = stream, backend

    def read(self, max_bytes, timeout=None):
        value = self.stream.read(max_bytes, self.backend.timeout(timeout))
        self.backend.timeout(timeout)
        return value

    def write(self, buffer, timeout=None):
        for start in range(0, len(buffer), 16384):
            self.stream.write(
                buffer[start : start + 16384], self.backend.timeout(timeout)
            )
            self.backend.timeout(timeout)

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        try:
            upgraded = self.stream.start_tls(
                ssl_context, server_hostname, self.backend.timeout(timeout)
            )
        except BaseException:
            self.close()
            raise
        with self.backend.lock:
            self.backend.streams.discard(self)
        return self.backend.wrap(upgraded)

    def get_extra_info(self, info):
        return self.stream.get_extra_info(info)

    def close(self):
        try:
            self.stream.close()
        finally:
            with self.backend.lock:
                self.backend.streams.discard(self)

    def abort(self):
        # shutdown interrupts a recv in another thread; close alone may wait for
        # that recv to finish. These sockets belong only to this tracing client.
        sock = self.stream.get_extra_info("socket")
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.close()
