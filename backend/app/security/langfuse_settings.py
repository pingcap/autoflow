"""Atomic, destination-bound Langfuse configuration with write-only credentials."""

import base64
from dataclasses import dataclass, field
import hashlib
import hmac
import json
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select, update

from app.core.config import settings
from app.models.site_setting import SiteSetting
from app.models.base import get_aes_key
from app.security.provider_config import endpoint_origin
from app.utils.aes import AESCipher

CONFIG_NAME = "langfuse_config_v1"
LEGACY_NAMES = {"langfuse_host", "langfuse_public_key", "langfuse_secret_key"}
DEFAULT_HOST = "https://us.cloud.langfuse.com"
PUBLIC_ORIGINS = {DEFAULT_HOST, "https://cloud.langfuse.com"}


class LangfuseConfigurationError(ValueError):
    """Only fixed, public-safe messages belong in this exception."""


class ConfigurationConflict(LangfuseConfigurationError):
    pass


class LangfuseUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    host: str = Field(max_length=2048)
    public_key: str = Field(max_length=512)
    secret_key: SecretStr | None = Field(default=None, max_length=4096)
    clear_secret: bool = False
    revision: str = Field(min_length=1, max_length=128)


@dataclass(frozen=True)
class RuntimeConfiguration:
    host: str
    public_key: str
    secret_key: str = field(repr=False)
    revision: str

    @property
    def enabled(self):
        return bool(self.public_key and self.secret_key)


def validate_host(host):
    try:
        origin = endpoint_origin(host)
    except ValueError:
        raise LangfuseConfigurationError("Invalid Langfuse host") from None
    allowed = PUBLIC_ORIGINS | {
        endpoint_origin(item) for item in settings.LANGFUSE_ALLOWED_ORIGINS
    }
    if origin not in allowed:
        raise LangfuseConfigurationError(
            "Langfuse host must be registered by the deployment operator"
        )
    # Preserve self-hosted base paths, but never accept credentials/query/fragment.
    return host.rstrip("/")


def _context(host, public_key):
    identity = json.dumps([host, public_key], separators=(",", ":"))
    return "site-settings.langfuse.v1:" + hashlib.sha256(identity.encode()).hexdigest()


def encrypt_secret(secret, host, public_key, *, key=None):
    if not secret:
        return ""
    cipher = AESCipher(key or get_aes_key(), context=_context(host, public_key))
    return base64.b64encode(cipher.encrypt(secret)).decode()


def decrypt_secret(record, *, key=None):
    if not record["secret_ciphertext"]:
        return ""
    try:
        cipher = AESCipher(
            key or get_aes_key(),
            context=_context(record["host"], record["public_key"]),
        )
        return cipher.decrypt(
            base64.b64decode(record["secret_ciphertext"], validate=True)
        )
    except (ValueError, TypeError):
        raise LangfuseConfigurationError(
            "Langfuse credential authentication failed"
        ) from None


def snapshot(values):
    record = values.get(CONFIG_NAME)
    if record is not None:
        if (
            not isinstance(record, dict)
            or record.get("version") != 1
            or any(
                not isinstance(record.get(k), str)
                for k in ("host", "public_key", "secret_ciphertext", "revision")
            )
        ):
            raise LangfuseConfigurationError("Invalid Langfuse configuration")
        return dict(record)
    legacy = {
        "host": values.get("langfuse_host") or DEFAULT_HOST,
        "public_key": values.get("langfuse_public_key") or "",
        "legacy_secret": values.get("langfuse_secret_key") or "",
    }
    if any(not isinstance(value, str) for value in legacy.values()):
        raise LangfuseConfigurationError("Invalid legacy Langfuse configuration")
    # A keyed revision cannot act as an offline verifier for a legacy plaintext key.
    revision = hmac.new(
        settings.SECRET_KEY.get_secret_value().encode(),
        b"langfuse-settings-revision-v1\0"
        + json.dumps(legacy, sort_keys=True).encode(),
        hashlib.sha256,
    ).hexdigest()
    return {"version": 0, "revision": revision, **legacy}


def _values(session, *, lock=False):
    query = select(SiteSetting).where(
        SiteSetting.name.in_(LEGACY_NAMES | {CONFIG_NAME})
    )
    if lock:
        query = query.with_for_update()
    rows = session.exec(query.execution_options(populate_existing=True)).all()
    return {row.name: row.value for row in rows}


def public_configuration(values):
    record = snapshot(values)
    configured = bool(record.get("secret_ciphertext") or record.get("legacy_secret"))
    return {
        "host": record["host"],
        "public_key": record["public_key"],
        "configured": configured,
        "revision": record["revision"],
        "migration_required": bool(record["version"] == 0 and configured),
    }


def read_public_configuration(session):
    return public_configuration(_values(session))


def read_runtime_configuration(session):
    record = snapshot(_values(session))
    if record["version"] == 0:
        if record["legacy_secret"]:
            raise LangfuseConfigurationError(
                "Run the Langfuse credential migration before starting workers"
            )
        secret = ""
    else:
        secret = decrypt_secret(record)
    if secret:
        validate_host(record["host"])
    return RuntimeConfiguration(
        record["host"], record["public_key"], secret, record["revision"]
    )


def save_configuration(session: Session, request: LangfuseUpdate, *, key=None):
    """Write within the caller's transaction; no network calls or internal commit."""
    values = _values(session, lock=True)
    current = snapshot(values)
    if not hmac.compare_digest(current["revision"], request.revision):
        raise ConfigurationConflict("Langfuse settings changed; reload before saving")
    host = validate_host(request.host)
    secret = (
        request.secret_key.get_secret_value()
        if request.secret_key is not None
        else None
    )
    if request.clear_secret and secret is not None:
        raise LangfuseConfigurationError(
            "Choose either replacement or removal of the secret"
        )
    if secret is not None and (not secret or not request.public_key):
        raise LangfuseConfigurationError(
            "A non-empty public key and secret key are required"
        )
    changed_destination = (host, request.public_key) != (
        current["host"],
        current["public_key"],
    )
    configured = bool(current.get("secret_ciphertext") or current.get("legacy_secret"))
    if configured and secret is None and not request.clear_secret:
        if changed_destination:
            raise LangfuseConfigurationError(
                "Changing the Langfuse destination requires fresh credentials"
            )
        if current["version"] == 0:
            raise LangfuseConfigurationError(
                "Migrate or replace the legacy Langfuse credential first"
            )
    ciphertext = (
        ""
        if request.clear_secret
        else encrypt_secret(secret, host, request.public_key, key=key)
        if secret is not None
        else current.get("secret_ciphertext", "")
    )
    record = {
        "version": 1,
        "revision": uuid4().hex,
        "host": host,
        "public_key": request.public_key,
        "secret_ciphertext": ciphertext,
    }
    if CONFIG_NAME in values:
        result = session.exec(
            update(SiteSetting)
            .where(
                SiteSetting.name == CONFIG_NAME,
                SiteSetting.value == values[CONFIG_NAME],
            )
            .values(value=record)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise ConfigurationConflict(
                "Langfuse settings changed; reload before saving"
            )
    else:
        session.add(SiteSetting(name=CONFIG_NAME, value=record, data_type="dict"))
        try:
            session.flush()
        except IntegrityError:
            raise ConfigurationConflict(
                "Langfuse settings changed; reload before saving"
            ) from None
    # Remove the legacy plaintext in the same transaction, including fresh replacements.
    session.exec(
        update(SiteSetting)
        .where(SiteSetting.name == "langfuse_secret_key")
        .values(value="")
    )
    return public_configuration({CONFIG_NAME: record})
