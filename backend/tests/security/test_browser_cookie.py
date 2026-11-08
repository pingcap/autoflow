from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api_server import identify_browser
from app.core.config import settings


def test_browser_identity_is_signed_http_only_secure_and_not_publicly_cached():
    app = FastAPI()
    app.middleware("http")(identify_browser)

    @app.get("/identity")
    def identity(request: Request):
        return {"browser_id": request.state.browser_id}

    with TestClient(app, base_url="https://testserver") as client:
        first = client.get("/identity")
        cookie = first.headers["set-cookie"]
        assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=lax" in cookie
        assert first.headers["cache-control"] == "private, no-store"
        browser_id = first.json()["browser_id"]
        assert client.get("/identity").json()["browser_id"] == browser_id
        client.cookies.clear()
        client.cookies.set(settings.BROWSER_ID_COOKIE_NAME, browser_id)
        assert client.get("/identity").json()["browser_id"] != browser_id


def test_legacy_cli_entry_point_uses_the_hardened_application():
    import main
    from app.api_server import app

    assert main.app is app
