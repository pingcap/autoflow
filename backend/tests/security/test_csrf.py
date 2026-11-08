import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from app.core.config import Settings
from app.security.csrf import CSRFMiddleware


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"Origin": "https://service.example"}, 200),
        ({"Origin": "https://app.example"}, 200),
        ({"Origin": "https://evil.example"}, 403),
        ({"Origin": "https://service.example.evil"}, 403),
        ({"Origin": "null"}, 403),
        ({"Origin": "http://service.example"}, 403),
        ({"Origin": "https://evil.example", "X-CSRF-Guard": "1"}, 403),
        ({"Origin": "https://evil.example", "Authorization": "Bearer fake"}, 403),
        ({"Referer": "https://service.example/page?private=query"}, 200),
        ({"Referer": "https://evil.example/page"}, 403),
        ({"X-CSRF-Guard": "1"}, 200),
        ({"Authorization": "Bearer client-key"}, 200),
        ({"X-CSRF-Guard": "1", "Sec-Fetch-Site": "cross-site"}, 403),
        ({"X-CSRF-Guard": "1", "Sec-Fetch-Site": "same-site"}, 403),
        ({}, 403),
    ],
)
def test_cookie_writes_require_verified_origin_or_explicit_nonbrowser_header(
    headers, expected
):
    app = FastAPI()
    mutations = []

    @app.post("/change")
    def change():
        mutations.append(True)
        return {"ok": True}

    app.add_middleware(
        CSRFMiddleware,
        allowed_origins=["https://app.example"],
        cookie_names=["session", "bid"],
    )
    with TestClient(app, base_url="https://service.example") as client:
        response = client.post(
            "/change", headers={"Cookie": "session=test-session", **headers}
        )
    assert response.status_code == expected
    assert bool(mutations) == (expected == 200)


def test_originless_stateless_clients_and_safe_reads_remain_supported():
    app = FastAPI()

    @app.api_route("/resource", methods=["GET", "POST"])
    def resource():
        return {}

    app.add_middleware(
        CSRFMiddleware, allowed_origins=[], cookie_names=["session", "bid"]
    )
    with TestClient(app) as client:
        assert client.post("/resource").status_code == 200
        assert (
            client.get(
                "/resource",
                headers={"Origin": "https://evil.example", "Cookie": "bid=test"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/resource", headers={"Origin": "https://evil.example"}
            ).status_code
            == 403
        )
        assert (
            client.post("/resource", headers={"Cookie": "bid=test"}).status_code == 403
        )


@pytest.mark.parametrize(
    "origins",
    [
        ["https://*.example.com"],
        ["https://user:password@example.com"],
        ["https://example.com/path"],
        ["https://example.com?token=private"],
        ["ftp://example.com"],
    ],
)
def test_credentialed_cors_rejects_nonexact_origins(origins):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, BACKEND_CORS_ORIGINS=origins)


def test_legacy_wildcard_cors_regex_cannot_be_enabled():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, BACKEND_CORS_ORIGIN_REGEXP=".*")


def test_e2e_browser_origin_can_login_through_container_proxy():
    from pathlib import Path
    from dotenv import dotenv_values

    # Read only the public allowlist, not deployment configuration or credentials.
    fixture = Path(__file__).resolve().parents[3] / "e2e/.env.backend"
    origins = dotenv_values(fixture)["BACKEND_CORS_ORIGINS"].split(",")
    app = FastAPI()

    @app.post("/api/v1/auth/login", status_code=204)
    def login():
        pass

    app.add_middleware(
        CSRFMiddleware, allowed_origins=origins, cookie_names=["session", "bid"]
    )
    with TestClient(app, base_url="http://backend:8000") as client:
        for origin, status in [
            ("http://127.0.0.1:3000", 204),
            ("http://localhost:3000", 204),
            ("http://127.0.0.1:3001", 403),
            ("http://localhost:3000.evil.example", 403),
        ]:
            response = client.post(
                "/api/v1/auth/login",
                headers={"Origin": origin, "Cookie": "bid=test-browser"},
            )
            assert response.status_code == status
