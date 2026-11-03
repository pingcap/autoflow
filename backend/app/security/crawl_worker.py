"""Isolated crawler entrypoint: stdin/stdout are bounded data, never application secrets."""

import json
import socket
import sys
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from app.security.crawl_html import IGNORE_TAGS, IGNORE_CLASSES
from app.security.crawl_http import CrawlHTTP, CrawlRejected, MAX_URLS, checked_url

MAX_TEXT_BYTES = 2 * 1024 * 1024


def sitemap_urls(url, private_origins=()):
    response = CrawlHTTP(private_origins).fetch(url)
    if response.status >= 400:
        raise CrawlRejected("Sitemap request failed")
    soup = BeautifulSoup(response.body, "html.parser")
    urls = []
    for loc in soup.find_all("loc", limit=MAX_URLS + 1):
        target = str(checked_url(urljoin(response.url, loc.get_text().strip())))
        if len(target) > 512:
            raise CrawlRejected("Document URL is too long")
        urls.append(target)
    if not urls or len(urls) > MAX_URLS:
        raise CrawlRejected("Sitemap must contain between 1 and 200 URLs")
    return urls


def render_page(url, private_origins=()):
    fetcher = CrawlHTTP(private_origins)
    initial = fetcher.fetch(url)
    if initial.status >= 400 or initial.headers.get("content-type", "").split(";")[
        0
    ].lower() not in {"text/html", "application/xhtml+xml", "text/plain"}:
        raise CrawlRejected("Crawl target did not return a document")
    final_url = initial.url
    cached = {str(checked_url(initial.url).copy_with(fragment=None)): initial}
    # Reserve a port without listening. Any browser request missed by interception
    # hits this unavailable proxy, including loopback destinations.
    with socket.socket() as unavailable_proxy, sync_playwright() as playwright:
        unavailable_proxy.bind(("127.0.0.1", 0))
        browser = playwright.chromium.launch(
            headless=True,
            proxy={
                "server": f"http://127.0.0.1:{unavailable_proxy.getsockname()[1]}",
                "bypass": "<-loopback>",
            },
            args=[
                "--disable-quic",
                "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                "--host-resolver-rules=MAP * ~NOTFOUND",
                "--js-flags=--max-old-space-size=128",
            ],
        )
        try:
            context = browser.new_context(
                service_workers="block", accept_downloads=False
            )
            # WebRTC is unnecessary for text imports and can open transports
            # outside HTTP routing. Remove its window entrypoints in every frame.
            context.add_init_script("""
                for (const name of ['RTCPeerConnection', 'webkitRTCPeerConnection']) {
                    Object.defineProperty(globalThis, name, {value: undefined, writable: false, configurable: false});
                }
            """)
            # Routed sockets are mocked unless connect_to_server() is called.
            # Never create a server connection for a crawling page.
            context.route_web_socket("**/*", lambda ws: None)
            page = context.new_page()
            context.on("page", lambda other: other.close() if other != page else None)

            def route_request(route):
                nonlocal final_url
                request = route.request
                try:
                    if request.method != "GET" or request.resource_type in {
                        "image",
                        "media",
                        "font",
                    }:
                        route.abort()
                        return
                    response = cached.pop(request.url, None) or fetcher.fetch(
                        request.url
                    )
                    if (
                        request.is_navigation_request()
                        and request.frame == page.main_frame
                    ):
                        final_url = response.url
                    headers = {
                        key: value
                        for key, value in response.headers.items()
                        if key
                        not in {
                            "content-length",
                            "content-encoding",
                            "transfer-encoding",
                            "connection",
                            "set-cookie",
                            "location",
                            "refresh",
                            "alt-svc",
                        }
                    }
                    route.fulfill(
                        status=response.status, headers=headers, body=response.body
                    )
                except Exception:
                    # Never fall back to browser networking when validation or IO fails.
                    route.abort()

            context.route("**/*", route_request)
            page.goto(
                initial.url,
                wait_until="domcontentloaded",
                timeout=max(1, fetcher.remaining() * 1000),
            )
            try:
                page.wait_for_load_state(
                    "networkidle", timeout=min(5000, max(1, fetcher.remaining() * 1000))
                )
            except PlaywrightTimeout:
                pass
            checked_url(page.url)
            if len(final_url) > 512:
                raise CrawlRejected("Document URL is too long")
            html = page.content()
            if len(html.encode("utf-8")) > MAX_TEXT_BYTES:
                raise CrawlRejected("Rendered document is too large")
            title = page.title()[:256]
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all(IGNORE_TAGS):
                tag.decompose()
            for class_name in IGNORE_CLASSES:
                for tag in soup.find_all(class_=class_name):
                    tag.decompose()
            content = MarkdownConverter().convert_soup(soup)
            if len(content.encode("utf-8")) > MAX_TEXT_BYTES:
                raise CrawlRejected("Rendered document is too large")
            return {"url": final_url, "title": title, "content": content}
        finally:
            browser.close()


if __name__ == "__main__":
    try:
        job = json.loads(sys.stdin.read(16384))
        operation = sitemap_urls if job["kind"] == "sitemap" else render_page
        result = {"result": operation(job["url"], job.get("private_origins", []))}
    except Exception:
        result = {"error": "Crawl failed or was blocked by its safety limits"}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
