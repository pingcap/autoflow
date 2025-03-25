import logging
from datetime import datetime, UTC
from typing import Generator

from autoflow.datasources.mime_types import SupportedMimeTypes
from autoflow.models import DBDocument


logger = logging.getLogger(__name__)


IGNORE_TAGS = [
    "noscript",
    "title",
    "script",
    "style",
    "meta",
    "head",
    "header",
    "footer",
    "nav",
    "symbol",
    "aside",
]

IGNORE_CLASSES = ["header", "footer", "sidebar"]


def load_web_documents(urls: list[str]) -> Generator[DBDocument, None, None]:
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup
    from markdownify import MarkdownConverter

    visited = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for url in urls:
            page = browser.new_page()
            response = page.goto(url)
            final_url = page.url
            if final_url in visited:
                continue

            if response.status >= 400:
                logger.error(
                    f"Failed to load page: {url}, response status: {response.status()}, skipping"
                )
                continue
            soup = BeautifulSoup(page.content(), "html.parser")
            for t in IGNORE_TAGS:
                for tag in soup.find_all(t):
                    tag.extract()
            for c in IGNORE_CLASSES:
                for tag in soup.find_all(class_=c):
                    tag.extract()
            content = MarkdownConverter().convert_soup(soup)
            title = page.title()
            visited.add(final_url)
            document = DBDocument(
                name=title,
                hash=hash(content),
                content=content,
                mime_type=SupportedMimeTypes.PLAIN_TXT,
                source_uri=final_url,
                last_modified_at=datetime.now(UTC),
            )
            yield document
        browser.close()
