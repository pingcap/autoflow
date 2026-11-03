import hashlib
import logging
import time
from datetime import datetime, UTC
from typing import Generator

from app.core.config import settings
from app.models import Document
from app.security.crawl_http import CrawlRejected, MAX_URLS
from app.security.crawler import crawl

logger = logging.getLogger(__name__)


def load_web_documents(
    knowledge_base_id: int, data_source_id: int, urls: list[str]
) -> Generator[Document, None, None]:
    if not 1 <= len(urls) <= MAX_URLS:
        raise CrawlRejected("A crawl must contain between 1 and 200 URLs")
    visited = set()
    failures = 0
    deadline = time.monotonic() + 600
    for url in urls:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CrawlRejected("Datasource crawl deadline exceeded")
        try:
            result = crawl(
                url,
                private_origins=settings.CRAWLER_ALLOWED_PRIVATE_ORIGINS,
                timeout=min(45, remaining),
            )
        except CrawlRejected:
            failures += 1
            logger.warning(
                "Crawl failed or was blocked for datasource %s", data_source_id
            )
            continue
        if result["url"] in visited:
            continue
        visited.add(result["url"])
        content = result["content"]
        yield Document(
            name=result["title"],
            hash=hashlib.sha256(content.encode()).hexdigest()[:32],
            content=content,
            mime_type="text/plain",
            knowledge_base_id=knowledge_base_id,
            data_source_id=data_source_id,
            source_uri=result["url"],
            last_modified_at=datetime.now(UTC),
        )
    if failures:
        raise CrawlRejected(f"{failures} crawl requests failed or were blocked")
