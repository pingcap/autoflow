from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, SecretStr, ValidationError

from app.api.deps import CurrentSuperuserDep, SessionDep
from app.security.langfuse_settings import (
    ConfigurationConflict,
    LangfuseConfigurationError,
    LangfuseUpdate,
    read_public_configuration,
    save_configuration,
    validate_host,
)
from app.security.langfuse_http import langfuse_http_client

router = APIRouter()


class LangfuseSetting(BaseModel):
    host: str = Field(default="https://us.cloud.langfuse.com", max_length=2048)
    public_key: str = Field(min_length=1, max_length=512)
    secret_key: SecretStr = Field(min_length=1, max_length=4096)


class LangfuseTestResult(BaseModel):
    success: bool
    error: str = ""


def validated(model, payload):
    try:
        return model.model_validate(payload)
    except ValidationError:
        # Default validation errors include submitted values, even secret fields.
        raise HTTPException(422, "Invalid Langfuse configuration") from None


@router.get("/admin/langfuse")
def get_langfuse(user: CurrentSuperuserDep, session: SessionDep):
    return read_public_configuration(session)


@router.put("/admin/langfuse")
def update_langfuse(
    user: CurrentSuperuserDep, session: SessionDep, payload: Any = Body()
):
    request = validated(LangfuseUpdate, payload)
    try:
        result = save_configuration(session, request)
        session.commit()
        return result
    except ConfigurationConflict as error:
        session.rollback()
        raise HTTPException(409, str(error)) from None
    except LangfuseConfigurationError as error:
        session.rollback()
        raise HTTPException(400, str(error)) from None


@router.post("/admin/langfuse/test")
def test_langfuse(
    user: CurrentSuperuserDep, payload: Any = Body()
) -> LangfuseTestResult:
    request = validated(LangfuseSetting, payload)
    try:
        host = validate_host(request.host)
        with langfuse_http_client(host) as client:
            response = client.get(
                host + "/api/public/projects",
                auth=(request.public_key, request.secret_key.get_secret_value()),
            )
            response.raise_for_status()
            data = response.json().get("data")
            success = isinstance(data, list) and bool(data)
    except Exception:
        # Never return bodies, URLs, authorization headers or exception text.
        success = False
    return LangfuseTestResult(
        success=success,
        error=""
        if success
        else "Langfuse verification failed; check credentials and the registered host.",
    )
