"""Operator-approved external chat and verification destinations."""

from app.core.config import settings
from app.security.provider_config import endpoint_origin, ProviderConfigurationError
from app.security.provider_http import ProviderSession


def validate_external_endpoint(value):
    if not value:
        return value
    allowed = {
        endpoint_origin(item) for item in settings.EXTERNAL_ENGINE_ALLOWED_ORIGINS
    }
    if endpoint_origin(value) not in allowed:
        raise ProviderConfigurationError(
            "External service endpoint is not registered in EXTERNAL_ENGINE_ALLOWED_ORIGINS"
        )
    return value


def external_engine_session(endpoint, *, verification=False):
    validate_external_endpoint(endpoint)
    return ProviderSession(
        endpoint,
        max_seconds=10 if verification else 120,
        max_response_bytes=65536 if verification else 4 * 1024 * 1024,
    )
