from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
import select
import threading
from types import SimpleNamespace

import httpcore
import pytest

from app.security.crawl_http import (
    CrawlHTTP,
    CrawlRejected,
    PinnedNetworkBackend,
    checked_url,
    public_address,
)
from app.security.crawler import crawl


@pytest.mark.parametrize(
    "value",
    [
        "file:///etc/passwd",
        "ftp://example.com",
        "http://user:secret@example.com",
        "http://example.com\\@localhost",
        "http://[fe80::1%25eth0]/",
        "http://example.com/\nsecret",
        "//example.com",
    ],
)
def test_crawler_rejects_non_http_credentials_and_ambiguous_urls(value):
    with pytest.raises(CrawlRejected):
        checked_url(value)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.100.100.200",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "::ffff:127.0.0.1",
        "fd00::1",
        "fe80::1",
        "64:ff9b::7f00:1",
        "2002:7f00:1::",
        "3fff::1",
    ],
)
def test_nonpublic_and_transition_addresses_are_rejected(address):
    assert not public_address(address)


def test_dns_is_validated_once_and_only_numeric_address_reaches_socket(monkeypatch):
    import app.security.crawl_http as target

    lookups, connections = [], []

    def lookup(host, **kwargs):
        lookups.append(host)
        return SimpleNamespace(addresses=lambda: iter(["8.8.8.8"]))

    def connect(self, host, port, **kwargs):
        connections.append((host, port))
        return SimpleNamespace(get_extra_info=lambda key: (host, port))

    monkeypatch.setattr(target.dns.resolver, "resolve_name", lookup)
    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", connect)
    PinnedNetworkBackend().connect_tcp("rebinding.example", 443)
    assert lookups == ["rebinding.example"] and connections == [("8.8.8.8", 443)]
    monkeypatch.setattr(
        target.dns.resolver,
        "resolve_name",
        lambda *args, **kwargs: SimpleNamespace(
            addresses=lambda: iter(["8.8.8.8", "127.0.0.1"])
        ),
    )
    with pytest.raises(CrawlRejected):
        PinnedNetworkBackend().connect_tcp("rebinding.example", 443)
    assert len(connections) == 1


@pytest.fixture
def website():
    requests = []
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, dict(self.headers)))
            body = b'<title>Dynamic page</title><main>Static content<script>fetch("/data").then(r=>r.text()).then(t=>document.querySelector("main").append(t));fetch("http://localhost:12345/private").catch(()=>{});new WebSocket("ws://localhost:12345/ws");</script></main>'
            body += b'<link rel="preconnect" href="http://localhost:12345"><script>navigator.serviceWorker.register("/sw.js").catch(()=>{});if(window.RTCPeerConnection){const pc=new RTCPeerConnection({iceServers:[{urls:"stun:127.0.0.1:UDP_PORT"}]});pc.createDataChannel("test");pc.createOffer().then(o=>{pc.setLocalDescription(o);fetch("/rtc-started")});}else{fetch("/rtc-blocked")}</script>'
            body = body.replace(b"UDP_PORT", str(udp.getsockname()[1]).encode())
            body = body.replace(
                b"localhost:12345", f"localhost:{private_server.server_port}".encode()
            )
            self.send_response(302 if self.path.startswith("/redirect") else 200)
            if self.path == "/redirect-safe":
                self.send_header("Location", "/page")
            if self.path == "/redirect-private":
                self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            if self.path == "/sitemap":
                body = b"<urlset><url><loc>/page</loc></url></urlset>"
            if self.path == "/data":
                body = b"Dynamic content"
            if self.path == "/loop":
                body = b"<script>while(true){}</script>"
            if self.path == "/large":
                self.send_header("Content-Length", str(6 * 1024 * 1024))
            if self.path == "/compressed":
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass

    class PrivateServer(ThreadingHTTPServer):
        def get_request(self):
            connection = super().get_request()
            requests.append(("/private-connection", {}))
            connection[0].settimeout(2)
            return connection

    private_server = PrivateServer(("127.0.0.1", 0), Handler)
    private_thread = threading.Thread(target=private_server.serve_forever, daemon=True)
    private_thread.start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    yield base, requests, udp
    server.shutdown()
    server.server_close()
    thread.join()
    private_server.shutdown()
    private_server.server_close()
    private_thread.join()
    udp.close()


def test_redirects_are_revalidated_and_response_budgets_enforced(website, monkeypatch):
    base, requests, udp = website
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    fetcher = CrawlHTTP([base])
    response = fetcher.fetch(base + "/redirect-safe")
    assert response.url == base + "/page" and response.status == 200
    for path in ["/redirect-private", "/large", "/compressed"]:
        with pytest.raises(CrawlRejected):
            fetcher.fetch(base + path)
    assert all(
        "Authorization" not in headers and "Cookie" not in headers
        for _, headers in requests
    )
    fetcher.requests = 100
    with pytest.raises(CrawlRejected):
        fetcher.fetch(base)
    with pytest.raises(CrawlRejected):
        CrawlHTTP([base], seconds=-1).fetch(base)


def test_real_browser_renders_javascript_and_isolated_sitemap_is_supported(website):
    base, requests, udp = website
    result = crawl(base + "/redirect-safe", private_origins=[base])
    assert result["title"] == "Dynamic page"
    assert "Dynamic content" in result["content"]
    assert result["url"] == base + "/page"
    assert crawl(base + "/sitemap", kind="sitemap", private_origins=[base]) == [
        base + "/page"
    ]
    assert all("/private" not in path and "/ws" not in path for path, _ in requests)
    assert not select.select([udp], [], [], 0)[0]
    assert any(path == "/rtc-blocked" for path, _ in requests)
    assert all(path != "/sw.js" for path, _ in requests)


def test_child_receives_no_application_secrets(monkeypatch):
    import app.security.crawler as target

    captured = {}
    monkeypatch.setenv("SECRET_KEY", "sensitive-signing-material")
    monkeypatch.setenv("OPENAI_API_KEY", "sensitive-provider-material")
    monkeypatch.setenv("HTTP_PROXY", "sensitive-proxy-material")

    class Child:
        returncode = 0

        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def communicate(self, *args, **kwargs):
            return json.dumps({"result": []}).encode(), None

        def poll(self):
            return 0

    monkeypatch.setattr(target.subprocess, "Popen", Child)
    assert crawl("https://example.com", kind="sitemap") == []
    assert not {"SECRET_KEY", "OPENAI_API_KEY", "HTTP_PROXY"} & captured["env"].keys()
    assert captured["start_new_session"]


def test_hung_renderer_is_killed_with_descendants(website):
    import psutil
    import time

    base, _, _ = website
    before = {
        process.pid for process in psutil.Process(os.getpid()).children(recursive=True)
    }
    with pytest.raises(CrawlRejected, match="deadline"):
        crawl(base + "/loop", private_origins=[base], timeout=4)
    time.sleep(0.2)
    after = {
        process.pid
        for process in psutil.Process(os.getpid()).children(recursive=True)
        if process.status() != psutil.STATUS_ZOMBIE
    }
    assert not after - before


def test_pinned_connection_preserves_tls_hostname_verification(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    import ssl
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import app.security.crawl_http as target

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "crawl.example")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("crawl.example")]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    certfile, keyfile = tmp_path / "certificate.pem", tmp_path / "key.pem"
    certfile.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    keyfile.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    seen_hosts, seen_sni = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen_hosts.append(self.headers["Host"])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"valid TLS")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)
    context.set_servername_callback(
        lambda sock, hostname, ctx: seen_sni.append(hostname)
    )
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    trusted = ssl.create_default_context(cafile=str(certfile))
    monkeypatch.setattr(target.ssl, "create_default_context", lambda: trusted)
    monkeypatch.setattr(
        target.dns.resolver,
        "resolve_name",
        lambda *args, **kwargs: SimpleNamespace(addresses=lambda: iter(["127.0.0.1"])),
    )
    base = f"https://crawl.example:{server.server_port}"
    wrong = f"https://wrong.example:{server.server_port}"
    try:
        assert CrawlHTTP([base]).fetch(base).body == b"valid TLS"
        assert seen_hosts == [f"crawl.example:{server.server_port}"]
        assert seen_sni == ["crawl.example"]
        with pytest.raises(httpcore.ConnectError):
            CrawlHTTP([wrong]).fetch(wrong)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
