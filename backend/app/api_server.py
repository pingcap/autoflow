import app.logger
import sentry_sdk

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware
from app.security.csrf import CSRFMiddleware
from app.security.generation import GenerationBodyLimitMiddleware
from app.security.audit import AuditMiddleware
from app.api.main import api_router
from app.core.config import settings
from app.site_settings import SiteSetting
from app.auth.browser_session import create_browser_session, verify_browser_session


load_dotenv()


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


if settings.SENTRY_DSN and settings.ENVIRONMENT != "local":
    sentry_sdk.init(
        dsn=str(settings.SENTRY_DSN),
        enable_tracing=True,
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        profiles_sample_rate=settings.SENTRY_PROFILES_SAMPLE_RATE,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    SiteSetting.update_db_cache()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
)


@app.middleware("http")
async def identify_browser(request: Request, call_next):
    browser_id = verify_browser_session(
        request.cookies.get(settings.BROWSER_ID_COOKIE_NAME),
        settings.SECRET_KEY.get_secret_value(),
        settings.BROWSER_ID_COOKIE_MAX_AGE,
    )
    has_browser_id = bool(browser_id)
    if not browser_id:
        browser_id, browser_cookie = create_browser_session(
            settings.SECRET_KEY.get_secret_value()
        )
    request.state.browser_id = browser_id
    response: Response = await call_next(request)
    if not has_browser_id:
        response.set_cookie(
            settings.BROWSER_ID_COOKIE_NAME,
            browser_cookie,
            max_age=settings.BROWSER_ID_COOKIE_MAX_AGE,
            secure=settings.SESSION_COOKIE_SECURE,
            httponly=True,
            samesite="lax",
        )
    response.headers["Cache-Control"] = "private, no-store"
    return response


app.include_router(api_router, prefix=settings.API_V1_STR)


# Origin checks run before browser cookies are created. CORS remains outermost so
# valid preflights do not create identities and trusted clients can read errors.
app.add_middleware(GenerationBodyLimitMiddleware, prefix=settings.API_V1_STR)
app.add_middleware(
    CSRFMiddleware,
    allowed_origins=[settings.server_host, *settings.BACKEND_CORS_ORIGINS],
    cookie_names=[settings.SESSION_COOKIE_NAME, settings.BROWSER_ID_COOKIE_NAME],
)
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            str(origin).rstrip("/") for origin in settings.BACKEND_CORS_ORIGINS
        ],
        allow_credentials=True,
        allow_methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-CSRF-Guard", "X-Request-ID"],
    )

app.add_middleware(AuditMiddleware, prefix=settings.API_V1_STR)
