import logging
from pydantic import BaseModel

from fastapi import APIRouter
from langfuse import Langfuse

from app.api.deps import CurrentSuperuserDep

router = APIRouter()
logger = logging.getLogger(__name__)


class LangfuseSetting(BaseModel):
    host: str = "https://us.cloud.langfuse.com"
    public_key: str
    secret_key: str


class LangfuseTestResult(BaseModel):
    success: bool
    error: str = ""


@router.post("/admin/langfuse/test")
def test_langfuse(
    user: CurrentSuperuserDep,
    request: LangfuseSetting,
) -> LangfuseTestResult:
    try:
        lf = Langfuse(
            host=request.host,
            secret_key=request.secret_key,
            public_key=request.public_key,
        )
        success = lf.auth_check()
        if not success:
            error = "Langfuse authentication failed, please check public_key, secret_key and host."
        else:
            error = ""
    except Exception as e:
        success = False
        error = str(e)
    return LangfuseTestResult(success=success, error=error)
