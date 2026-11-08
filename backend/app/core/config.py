import enum
import base64
import hashlib
from typing import Annotated, Any
from urllib.parse import quote, urlsplit

from pydantic import (
    AnyUrl,
    BeforeValidator,
    Field,
    HttpUrl,
    MySQLDsn,
    SecretStr,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core import MultiHostUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self
from app.security.generation_limits import GenerationLimits


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Environment(str, enum.Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
        hide_input_in_errors=True,
    )
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: SecretStr
    DATA_ENCRYPTION_KEY: SecretStr
    DOMAIN: str = "localhost"
    ENVIRONMENT: Environment = Environment.LOCAL
    LOG_LEVEL: str = "INFO"
    SQLALCHEMY_LOG_LEVEL: str = "WARNING"

    SESSION_COOKIE_NAME: str = "session"
    # 90 days
    SESSION_COOKIE_MAX_AGE: int = Field(
        default=3600 * 24 * 90, ge=60, le=3600 * 24 * 90
    )
    SESSION_COOKIE_SECURE: bool = True

    BROWSER_ID_COOKIE_NAME: str = "bid"
    BROWSER_ID_COOKIE_MAX_AGE: int = Field(
        default=3600 * 24 * 14, ge=60, le=3600 * 24 * 90
    )

    @computed_field  # type: ignore[misc]
    @property
    def server_host(self) -> str:
        # Use HTTPS for anything other than local development
        if self.ENVIRONMENT == Environment.LOCAL:
            return f"http://{self.DOMAIN}"
        return f"https://{self.DOMAIN}"

    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []
    BACKEND_CORS_ORIGIN_REGEXP: str | None = None

    @field_validator("BACKEND_CORS_ORIGINS")
    @classmethod
    def cors_origins_must_be_exact(cls, values):
        import json
        from app.security.csrf import canonical_origin

        if isinstance(values, str):
            values = json.loads(values)
        if not isinstance(values, list) or len(values) > 100:
            raise ValueError("CORS requires a bounded list of exact origins")
        return [AnyUrl(canonical_origin(str(value))) for value in values]

    @field_validator("BACKEND_CORS_ORIGIN_REGEXP")
    @classmethod
    def reject_cors_regex(cls, value):
        if value:
            raise ValueError(
                "Use exact BACKEND_CORS_ORIGINS instead of a regular expression"
            )
        return None

    # Exact additional origins trusted by the deployment operator, never by API input.
    PROVIDER_ALLOWED_ORIGINS: list[str] = []
    LANGFUSE_ALLOWED_ORIGINS: list[str] = []
    CRAWLER_ALLOWED_PRIVATE_ORIGINS: list[str] = []

    @field_validator(
        "PROVIDER_ALLOWED_ORIGINS",
        "CRAWLER_ALLOWED_PRIVATE_ORIGINS",
        "LANGFUSE_ALLOWED_ORIGINS",
    )
    @classmethod
    def provider_origins_must_be_exact(cls, values):
        if len(values) > 100:
            raise ValueError("Too many provider origins")
        for value in values:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
                or "*" in value
                or "\\" in value
            ):
                raise ValueError(
                    "Provider origins must be exact HTTP(S) origins without paths or credentials"
                )
        return values

    PROJECT_NAME: str = "TiDB.AI"
    SENTRY_DSN: HttpUrl | None = None
    SENTRY_TRACES_SAMPLE_RATE: float = 1.0
    # Kept for configuration compatibility. Profiles remain disabled until their
    # separate envelope has a verified privacy boundary (see security/telemetry).
    SENTRY_PROFILES_SAMPLE_RATE: float = 0.0

    @model_validator(mode="after")
    def _validate_sentry_sample_rate(self) -> Self:
        if not self.SENTRY_DSN:
            return self
        if self.SENTRY_TRACES_SAMPLE_RATE < 0 or self.SENTRY_TRACES_SAMPLE_RATE > 1:
            raise ValueError("SENTRY_TRACES_SAMPLE_RATE must be between 0 and 1")
        if self.SENTRY_PROFILES_SAMPLE_RATE < 0 or self.SENTRY_PROFILES_SAMPLE_RATE > 1:
            raise ValueError("SENTRY_PROFILES_SAMPLE_RATE must be between 0 and 1")
        return self

    LOCAL_FILE_STORAGE_PATH: str = "/shared/data"

    TIDB_HOST: str = "127.0.0.1"
    TIDB_PORT: int = 4000
    TIDB_USER: str = "root"
    TIDB_PASSWORD: SecretStr = SecretStr("")
    TIDB_DATABASE: str
    TIDB_SSL: bool = True

    ENABLE_QUESTION_CACHE: bool = False

    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    GENERATION_LIMITS: GenerationLimits = Field(default_factory=GenerationLimits)
    GENERATION_REDIS_URL: SecretStr | None = None
    GENERATION_QUOTA_NAMESPACE: str = Field(
        default="autoflow", pattern=r"^[a-zA-Z0-9_-]{1,32}$"
    )
    GENERATION_TRUSTED_PROXY_CIDRS: list[str] = Field(default_factory=list)

    @field_validator("GENERATION_TRUSTED_PROXY_CIDRS")
    @classmethod
    def validate_generation_proxies(cls, values):
        import ipaddress

        if len(values) > 32:
            raise ValueError("Too many trusted proxies")
        for value in values:
            network = ipaddress.ip_network(value, strict=True)
            if network.prefixlen == 0:
                raise ValueError("A trusted proxy must not include every address")
        return values

    # TODO: move below config to `option` table, it should be configurable by staff in console
    TIDB_AI_CHAT_ENDPOINT: str = "https://tidb.ai/api/v1/chats"
    TIDB_AI_API_KEY: SecretStr | None = None

    COMPLIED_INTENT_ANALYSIS_PROGRAM_PATH: str | None = None
    COMPLIED_PREREQUISITE_ANALYSIS_PROGRAM_PATH: str | None = None

    # NOTICE: EMBEDDING_DIMS and EMBEDDING_MAX_TOKENS is deprecated and
    # will be removed in the future.
    EMBEDDING_DIMS: int = 1536
    EMBEDDING_MAX_TOKENS: int = 2048

    EVALUATION_OPENAI_API_KEY: SecretStr | None = None

    @computed_field(repr=False)  # type: ignore[misc]
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> MySQLDsn:
        return MultiHostUrl.build(
            scheme="mysql+pymysql",
            username=self.TIDB_USER,
            # TODO: remove quote after following issue is fixed:
            # https://github.com/pydantic/pydantic/issues/8061
            password=quote(self.TIDB_PASSWORD.get_secret_value()),
            host=self.TIDB_HOST,
            port=self.TIDB_PORT,
            path=self.TIDB_DATABASE,
            query="ssl_verify_cert=true&ssl_verify_identity=true"
            if self.TIDB_SSL
            else None,
        )

    @computed_field(repr=False)  # type: ignore[misc]
    @property
    def SQLALCHEMY_ASYNC_DATABASE_URI(self) -> MySQLDsn:
        return MultiHostUrl.build(
            scheme="mysql+asyncmy",
            username=self.TIDB_USER,
            password=quote(self.TIDB_PASSWORD.get_secret_value()),
            host=self.TIDB_HOST,
            port=self.TIDB_PORT,
            path=self.TIDB_DATABASE,
        )

    @field_serializer("SQLALCHEMY_DATABASE_URI", "SQLALCHEMY_ASYNC_DATABASE_URI")
    def redact_database_uri(self, value):
        return "<redacted database connection>"

    @model_validator(mode="after")
    def _validate_secrets(self) -> Self:
        if self.ENVIRONMENT != Environment.LOCAL and not self.SESSION_COOKIE_SECURE:
            raise ValueError("Secure cookies are required outside local development.")
        secret = self.SECRET_KEY.get_secret_value()
        if not secret:
            raise ValueError(
                "Please set a secret key using the SECRET_KEY environment variable."
            )

        if (
            hashlib.sha256(secret.encode()).hexdigest()
            == "ddb3bf032f1cf4833b5fdfbdd6dba219974500c3fdce7f57586874018a8fac6a"
        ):
            raise ValueError("The retired example signing key must be replaced")
        try:
            encryption_key = base64.b64decode(
                self.DATA_ENCRYPTION_KEY.get_secret_value(), validate=True
            )
        except (ValueError, TypeError):
            raise ValueError(
                "DATA_ENCRYPTION_KEY must be a base64-encoded 32-byte key"
            ) from None
        if len(encryption_key) != 32:
            raise ValueError(
                "DATA_ENCRYPTION_KEY must contain 32 securely generated bytes"
            )
        if encryption_key == secret.encode()[:32]:
            raise ValueError("Use separate signing and data-encryption keys")

        min_length = 32
        if len(secret.encode()) < min_length:
            message = (
                "The SECRET_KEY is too short, "
                f"please use a longer secret, at least {min_length} characters."
            )
            raise ValueError(message)
        return self


settings = Settings()  # type: ignore
