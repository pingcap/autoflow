"""Authenticated, expiring ownership for anonymous browser sessions."""

import base64
import hashlib
import hmac
import re
import time
from uuid import uuid4


def _signature(payload: str, secret: str) -> str:
    digest = hmac.new(
        secret.encode(), b"autoflow-browser-v1:" + payload.encode(), hashlib.sha256
    ).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def create_browser_session(secret: str, now: int | None = None) -> tuple[str, str]:
    browser_id = uuid4().hex
    payload = f"{browser_id}.{int(time.time()) if now is None else now}"
    return browser_id, f"v1.{payload}.{_signature(payload, secret)}"


def verify_browser_session(
    cookie: str | None, secret: str, max_age: int, now: int | None = None
) -> str | None:
    if not cookie or len(cookie) > 128:
        return None
    match = re.fullmatch(
        r"v1\.([0-9a-f]{32})\.([0-9]{1,12})\.([A-Za-z0-9_-]{43})", cookie
    )
    if not match:
        return None
    browser_id, issued, signature = match.groups()
    current = int(time.time()) if now is None else now
    if int(issued) > current + 60 or current - int(issued) >= max_age:
        return None
    if not hmac.compare_digest(signature, _signature(f"{browser_id}.{issued}", secret)):
        return None
    return browser_id
