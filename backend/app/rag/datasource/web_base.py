import logging
from datetime import datetime, UTC
from typing import Generator
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

from app.models import Document
from app.rag.datasource.consts import IGNORE_TAGS, IGNORE_CLASSES

logger = logging.getLogger(__name__)


def load_web_documents(
    knowledge_base_id: int, data_source_id: int, urls: list[str]
) -> Generator[Document, None, None]:
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
            document = Document(
                name=title,
                hash=hash(content),
                content=content,
                mime_type="text/plain",
                knowledge_base_id=knowledge_base_id,
                data_source_id=data_source_id,
                source_uri=final_url,
                last_modified_at=datetime.now(UTC),
            )
            yield document
        browser.close()
