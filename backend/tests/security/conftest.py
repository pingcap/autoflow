"""Isolated security tests must never use deployment credentials or services."""

import os
import secrets

os.environ.update(
    SECRET_KEY=secrets.token_urlsafe(48),
    ENVIRONMENT="local",
    TIDB_HOST="127.0.0.1",
    TIDB_DATABASE="autoflow_security_test",
    TIDB_USER="security_test",
    TIDB_PASSWORD="unused-test-password",
    TIDB_SSL="false",
    SENTRY_DSN="",
    DEEPEVAL_TELEMETRY_OPT_OUT="YES",
    OTEL_SDK_DISABLED="true",
    LITELLM_LOCAL_MODEL_COST_MAP="True",
)
