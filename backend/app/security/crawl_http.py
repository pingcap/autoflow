"""Credential-free crawler HTTP with bounded reads and DNS pinned at connection time."""

from dataclasses import dataclass
import ipaddress
import socket
import ssl
import time
from urllib.parse import urljoin

import dns.resolver
import httpcore
import httpx

MAX_URLS = 200
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_PAGE_BYTES = 20 * 1024 * 1024
MAX_REQUESTS = 100
SPECIAL_V4 = tuple(
    ipaddress.ip_network(value) for value in ("192.0.0.0/24", "192.88.99.0/24")
)
SPECIAL_V6 = tuple(
    ipaddress.ip_network(value)
    for value in (
        "64:ff9b::/96",
        "64:ff9b:1::/48",
        "2001::/23",
        "2002::/16",
        "3fff::/20",
    )
)


class CrawlRejected(ValueError):
    pass


def checked_url(value: str) -> httpx.URL:
    if (
        not isinstance(value, str)
        or len(value) > 2048
        or any(ord(c) < 33 or c == "\\" for c in value)
    ):
        raise CrawlRejected("Invalid crawl URL")
    try:
        url = httpx.URL(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.userinfo
            or "%" in url.host
            or len(str(url)) > 2048
        ):
            raise ValueError()
        return url
    except (ValueError, httpx.InvalidURL):
        raise CrawlRejected(
            "Only absolute HTTP(S) URLs without credentials are supported"
        ) from None


def origin(url: httpx.URL) -> str:
    return str(url.copy_with(path="/", query=None, fragment=None)).rstrip("/")


def public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv4Address) and any(
        address in network for network in SPECIAL_V4
    ):
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped:
            return public_address(str(address.ipv4_mapped))
        if any(address in network for network in SPECIAL_V6):
            return False
    return address.is_global and not (
        address.is_reserved
        or address.is_multicast
        or address.is_loopback
        or address.is_link_local
        or address.is_unspecified
    )


class PinnedNetworkBackend(httpcore.SyncBackend):
    def __init__(self, allow_private=False):
        self.allow_private = allow_private

    def connect_tcp(
        self, host, port, timeout=None, local_address=None, socket_options=None
    ):
        started = time.monotonic()
        timeout = min(timeout or 5, 5)
        try:
            addresses = [str(ipaddress.ip_address(host))]
        except ValueError:
            try:
                addresses = list(
                    dns.resolver.resolve_name(
                        host, family=socket.AF_UNSPEC, lifetime=min(3, timeout)
                    ).addresses()
                )
            except dns.exception.DNSException:
                raise CrawlRejected("Crawler DNS lookup failed") from None
        if (
            not addresses
            or len(addresses) > 32
            or (
                not self.allow_private
                and not all(public_address(value) for value in addresses)
            )
        ):
            raise CrawlRejected("Crawl target is not a public address")
        # Only numeric addresses reach the socket backend; the hostname is never
        # resolved again. HTTPCore retains the original hostname for TLS/SNI.
        for candidate in sorted(
            addresses, key=lambda value: ipaddress.ip_address(value).version
        ):
            address = str(ipaddress.ip_address(candidate))
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                break
            try:
                stream = super().connect_tcp(
                    address,
                    port,
                    timeout=min(2, remaining),
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout):
                continue
            peer = stream.get_extra_info("server_addr")
            if not peer or ipaddress.ip_address(peer[0]) != ipaddress.ip_address(
                address
            ):
                stream.close()
                raise CrawlRejected("Crawler peer address changed")
            return stream
        raise CrawlRejected("Crawler connection failed or exceeded its deadline")


@dataclass
class CrawlResponse:
    url: str
    status: int
    headers: dict[str, str]
    body: bytes


class CrawlHTTP:
    def __init__(self, private_origins=(), seconds=30):
        self.private_origins = {origin(checked_url(value)) for value in private_origins}
        self.deadline = time.monotonic() + seconds
        self.requests = 0
        self.bytes = 0

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise CrawlRejected("Crawler page deadline exceeded")
        return remaining

    def fetch(self, value: str) -> CrawlResponse:
        for _ in range(6):
            url = checked_url(value)
            allowed = origin(url) in self.private_origins
            if not allowed and url.port not in {None, 80, 443}:
                raise CrawlRejected("Crawler port is not permitted")
            self.requests += 1
            if self.requests > MAX_REQUESTS:
                raise CrawlRejected("Crawler request budget exceeded")
            timeout = min(5, self.remaining())
            with httpcore.ConnectionPool(
                ssl_context=ssl.create_default_context(),
                network_backend=PinnedNetworkBackend(allow_private=allowed),
                max_connections=1,
                max_keepalive_connections=0,
                retries=0,
            ) as pool:
                with pool.stream(
                    "GET",
                    str(url.copy_with(fragment=None)),
                    headers={
                        "User-Agent": "Autoflow-Crawler/1.0",
                        "Accept-Encoding": "identity",
                        "Accept": "text/html,application/xhtml+xml,application/xml,application/json,*/*;q=0.8",
                    },
                    extensions={
                        "timeout": dict.fromkeys(
                            ("connect", "read", "write", "pool"), timeout
                        )
                    },
                ) as response:
                    headers = {
                        key.decode("ascii").lower(): val.decode("latin1")
                        for key, val in response.headers
                    }
                    self.remaining()
                    if response.status in {301, 302, 303, 307, 308}:
                        if "location" not in headers:
                            raise CrawlRejected("Crawler redirect has no target")
                        value = urljoin(str(url), headers["location"])
                        continue
                    if (
                        headers.get("content-encoding", "identity").lower()
                        != "identity"
                    ):
                        raise CrawlRejected("Crawler requires an uncompressed response")
                    if int(headers.get("content-length", "0")) > MAX_RESPONSE_BYTES:
                        raise CrawlRejected("Crawler response is too large")
                    chunks, size = [], 0
                    for chunk in response.iter_stream():
                        self.remaining()
                        size += len(chunk)
                        self.bytes += len(chunk)
                        if size > MAX_RESPONSE_BYTES or self.bytes > MAX_PAGE_BYTES:
                            raise CrawlRejected("Crawler response budget exceeded")
                        chunks.append(chunk)
                    return CrawlResponse(
                        str(url), response.status, headers, b"".join(chunks)
                    )
        raise CrawlRejected("Too many crawler redirects")
