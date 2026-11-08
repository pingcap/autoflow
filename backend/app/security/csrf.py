"""Origin checks for browser writes, including same-site hostile subdomains."""

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse


def canonical_origin(value: str, *, allow_path=False) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 4096
        or any(ord(char) < 33 or char == "\\" for char in value)
    ):
        raise ValueError("Invalid origin")
    try:
        url = httpx.URL(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.userinfo
            or "*" in url.host
        ):
            raise ValueError("Invalid origin")
        if not allow_path and (url.path not in {"", "/"} or url.query or url.fragment):
            raise ValueError("Origin must not contain a path, query or fragment")
        return str(url.copy_with(path="/", query=None, fragment=None)).rstrip("/")
    except httpx.InvalidURL:
        raise ValueError("Invalid origin") from None


class CSRFMiddleware:
    def __init__(self, app, allowed_origins, cookie_names):
        self.app = app
        self.allowed_origins = {
            canonical_origin(str(value)) for value in allowed_origins
        }
        self.cookie_names = tuple(cookie_names)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in {"GET", "HEAD", "OPTIONS"}:
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        allowed = False
        try:
            trusted = self.allowed_origins | {
                canonical_origin(str(request.url), allow_path=True)
            }
            source = request.headers.get("origin")
            if source is not None:
                allowed = canonical_origin(source) in trusted
            elif request.headers.get("referer"):
                allowed = (
                    canonical_origin(request.headers["referer"], allow_path=True)
                    in trusted
                )
            else:
                has_cookie = any(
                    request.cookies.get(name) for name in self.cookie_names
                )
                # A custom header requires a successful CORS preflight in a browser;
                # it also permits explicit non-browser cookie clients and RSC calls.
                allowed = (
                    not has_cookie
                    or request.headers.get("x-csrf-guard") == "1"
                    or bool(request.headers.get("authorization"))
                )
                if request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}:
                    allowed = False
        except ValueError:
            allowed = False
        if not allowed:
            await JSONResponse(
                {"detail": "Request origin verification failed"}, status_code=403
            )(scope, receive, send)
            return
        await self.app(scope, receive, send)
