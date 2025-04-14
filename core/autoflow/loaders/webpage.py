import logging
from datetime import datetime, UTC
from typing import Generator, Optional, List
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

from autoflow.loaders.base import Loader
from autoflow.storage.doc_store import Document
from autoflow.data_types import DataType

logger = logging.getLogger(__name__)

# Common tags and classes to ignore when processing web content
IGNORE_TAGS = ["nav", "header", "footer", "script", "style", "noscript", "iframe"]
IGNORE_CLASSES = [
    "nav",
    "navigation",
    "footer",
    "header",
    "sidebar",
    "menu",
    "ad",
    "advertisement",
]


class WebpageLoader(Loader):
    def __init__(
        self,
        ignore_tags: Optional[List[str]] = None,
        ignore_classes: Optional[List[str]] = None,
    ):
        super().__init__()
        self._ignore_tags = ignore_tags or IGNORE_TAGS
        self._ignore_classes = ignore_classes or IGNORE_CLASSES

    def load(self, urls: str | list[str], **kwargs) -> Generator[Document, None, None]:
        if isinstance(urls, str):
            urls = [urls]

        visited = set()
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                for url in urls:
                    try:
                        page = browser.new_page()
                        response = page.goto(url)
                        final_url = page.url

                        if final_url in visited:
                            continue

                        if response is None or response.status >= 400:
                            logger.error(
                                f"Failed to load page: {url}, response status: {response.status if response else 'None'}, skipping"
                            )
                            continue

                        # Parse the content
                        soup = BeautifulSoup(page.content(), "html.parser")

                        # Remove unwanted elements
                        for tag in self._ignore_tags:
                            for element in soup.find_all(tag):
                                element.extract()

                        for class_name in self._ignore_classes:
                            for element in soup.find_all(class_=class_name):
                                element.extract()

                        # Convert to markdown
                        content = MarkdownConverter().convert_soup(soup)
                        title = page.title() or final_url

                        visited.add(final_url)

                        # Create document
                        document = Document(
                            name=title,
                            content=content,
                            data_type=DataType.HTML,
                            meta={
                                "source_uri": final_url,
                                "original_uri": url,
                                "last_modified": datetime.now(UTC).isoformat(),
                            },
                        )

                        yield document

                    except Exception as e:
                        logger.error(f"Error processing URL {url}: {str(e)}")
                        continue
                    finally:
                        if "page" in locals():
                            page.close()
            finally:
                browser.close()
