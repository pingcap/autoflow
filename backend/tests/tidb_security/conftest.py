"""No deployment dotenv, credentials or database URL may enter these tests."""

import base64
import os
import secrets
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlmodel import create_engine

port = os.environ.get("SECURITY_TIDB_TEST_PORT", "")
if (Path.cwd() / ".env").exists():
    raise RuntimeError("Deployment dotenv files are not allowed in these tests")
if not port.isdigit() or not 1 <= int(port) <= 65535:
    raise RuntimeError(
        "Use scripts/test_security_tidb.py to start an owned test database"
    )
os.environ.update(
    SECRET_KEY=secrets.token_urlsafe(48),
    DATA_ENCRYPTION_KEY=base64.b64encode(secrets.token_bytes(32)).decode(),
    ENVIRONMENT="local",
    TIDB_HOST="127.0.0.1",
    TIDB_PORT=port,
    TIDB_DATABASE="unused_security_fixture",
    TIDB_USER="root",
    TIDB_PASSWORD="",
    TIDB_SSL="false",
    SENTRY_DSN="",
    DEEPEVAL_TELEMETRY_OPT_OUT="YES",
    OTEL_SDK_DISABLED="true",
    LITELLM_LOCAL_MODEL_COST_MAP="True",
)


@pytest.fixture(scope="session", autouse=True)
def env():
    # Explicitly override the ordinary test suite's deployment-dotenv fixture.
    pass


@pytest.fixture
def database(monkeypatch):
    from app.core.config import settings

    name = "autoflow_security_" + uuid4().hex
    base = f"mysql+pymysql://root@127.0.0.1:{port}"
    admin = create_engine(
        base,
        connect_args={"connect_timeout": 3, "read_timeout": 10, "write_timeout": 10},
    )
    engine = None
    try:
        with admin.begin() as connection:
            connection.execute(text(f"CREATE DATABASE `{name}`"))
        engine = create_engine(
            base + "/" + name,
            connect_args={
                "connect_timeout": 3,
                "read_timeout": 10,
                "write_timeout": 10,
            },
        )
        # Exercise the shipped DDL, including timestamp precision and defaults;
        # ORM create_all is not the schema installation path used in production.
        monkeypatch.setattr(settings, "TIDB_DATABASE", name)
        # Some historical migrations import the global application engine.
        # A fresh interpreter prevents it retaining a previous fixture's DB.
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            env=dict(os.environ, TIDB_DATABASE=name),
            check=True,
            timeout=60,
        )
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS `{name}`"))
        admin.dispose()
