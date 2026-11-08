import asyncio
from contextlib import asynccontextmanager
import hmac
import logging
import os
import re
from threading import BoundedSemaphore, Lock
from typing import Annotated

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.responses import JSONResponse

try:
    from .model_manifest import EMBEDDING_MODELS, RERANKER_MODELS
except ImportError:
    from model_manifest import EMBEDDING_MODELS, RERANKER_MODELS

logger = logging.getLogger(__name__)
DEFAULT_EMBEDDING_MODEL = next(iter(EMBEDDING_MODELS))
DEFAULT_RERANKER_MODEL = next(iter(RERANKER_MODELS))
MAX_BODY_BYTES = 1024 * 1024
MAX_TOTAL_CHARACTERS = 65536
EMBEDDING_MODEL_DICT = {}
RERANKER_MODEL_DICT = {}
MODEL_LOCK = Lock()
INFERENCE_SLOT = BoundedSemaphore(1)


def authentication_token():
    token = os.environ.get("SIDECAR_AUTH_TOKEN", "")
    if not re.fullmatch(r"[A-Za-z0-9._~+/-]{32,128}={0,2}", token):
        raise RuntimeError(
            "Configure a strong SIDECAR_AUTH_TOKEN before starting the model service"
        )
    return token


class RequestBoundary:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] != "/api/v1/healthz":
            headers = dict(scope.get("headers", []))
            expected = ("Bearer " + authentication_token()).encode()
            if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
                return await JSONResponse(
                    {"detail": "Authentication required"}, status_code=401
                )(scope, receive, send)
        body = bytearray()
        try:
            async with asyncio.timeout(15):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > MAX_BODY_BYTES:
                        return await JSONResponse(
                            {"detail": "Request too large"}, status_code=413
                        )(scope, receive, send)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            return await JSONResponse({"detail": "Request timed out"}, status_code=408)(
                scope, receive, send
            )
        consumed = False

        async def bounded_receive():
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, bounded_receive, send)


def get_embedding_model(model_name):
    if model_name not in EMBEDDING_MODELS:
        raise HTTPException(400, "Embedding model is not approved")
    with MODEL_LOCK:
        if model_name not in EMBEDDING_MODEL_DICT:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                model_name_or_path=model_name,
                revision=EMBEDDING_MODELS[model_name],
                trust_remote_code=False,
                local_files_only=True,
            )
            model.max_seq_length = 2048
            EMBEDDING_MODEL_DICT[model_name] = model
        return EMBEDDING_MODEL_DICT[model_name]


def get_reranker_model(model_name):
    if model_name not in RERANKER_MODELS:
        raise HTTPException(400, "Reranker model is not approved")
    with MODEL_LOCK:
        if model_name not in RERANKER_MODEL_DICT:
            from sentence_transformers import CrossEncoder

            RERANKER_MODEL_DICT[model_name] = CrossEncoder(
                model_name_or_path=model_name,
                revision=RERANKER_MODELS[model_name],
                model_kwargs={"torch_dtype": "auto", "use_safetensors": True},
                trust_remote_code=False,
                local_files_only=True,
                max_length=2048,
            )
        return RERANKER_MODEL_DICT[model_name]


Text = Annotated[str, Field(min_length=1, max_length=8192)]
Texts = Annotated[list[Text], Field(min_length=1, max_length=32)]


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentences: Texts
    model: str = Field(default=DEFAULT_EMBEDDING_MODEL, max_length=128)
    normalize_embeddings: bool = True

    @model_validator(mode="after")
    def bounded(self):
        if sum(map(len, self.sentences)) > MAX_TOTAL_CHARACTERS:
            raise ValueError("Too much embedding input")
        if self.model not in EMBEDDING_MODELS:
            raise ValueError("Embedding model is not approved")
        return self


class EmbeddingResponse(BaseModel):
    model: str
    embeddings: list[list[float]]


class RerankerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(default=DEFAULT_RERANKER_MODEL, max_length=128)
    query: Text
    passages: Texts

    @model_validator(mode="after")
    def bounded(self):
        if len(self.query) + sum(map(len, self.passages)) > MAX_TOTAL_CHARACTERS:
            raise ValueError("Too much reranker input")
        if self.model not in RERANKER_MODELS:
            raise ValueError("Reranker model is not approved")
        return self


class RerankerResponse(BaseModel):
    model: str
    scores: list[float]


router = APIRouter()


@router.get("/healthz")
def healthz():
    return "OK"


@router.post("/embedding")
def get_texts_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    if not INFERENCE_SLOT.acquire(blocking=False):
        raise HTTPException(429, "Model service is busy", headers={"Retry-After": "1"})
    try:
        model = get_embedding_model(request.model)
        result = model.encode(
            request.sentences,
            batch_size=4,
            show_progress_bar=False,
            normalize_embeddings=request.normalize_embeddings,
        )
        return EmbeddingResponse(model=request.model, embeddings=result.tolist())
    except HTTPException:
        raise
    except Exception as error:
        logger.error(
            "Local model inference failed", extra={"error_type": type(error).__name__}
        )
        raise HTTPException(503, "Local model inference is unavailable") from None
    finally:
        INFERENCE_SLOT.release()


@router.post("/reranker")
def reranker_texts(request: RerankerRequest) -> RerankerResponse:
    if not INFERENCE_SLOT.acquire(blocking=False):
        raise HTTPException(429, "Model service is busy", headers={"Retry-After": "1"})
    try:
        model = get_reranker_model(request.model)
        result = model.predict(
            [(request.query, passage) for passage in request.passages],
            batch_size=4,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        return RerankerResponse(model=request.model, scores=result.tolist())
    except HTTPException:
        raise
    except Exception as error:
        logger.error(
            "Local model inference failed", extra={"error_type": type(error).__name__}
        )
        raise HTTPException(503, "Local model inference is unavailable") from None
    finally:
        INFERENCE_SLOT.release()


@asynccontextmanager
async def lifespan(app):
    authentication_token()
    if os.environ.get("PRE_LOAD_DEFAULT_EMBEDDING_MODEL", "true").lower() == "true":
        get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    if os.environ.get("PRE_LOAD_DEFAULT_RERANKER_MODEL", "false").lower() == "true":
        get_reranker_model(DEFAULT_RERANKER_MODEL)
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(RequestBoundary)
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5001,
        workers=1,
        limit_concurrency=16,
        timeout_keep_alive=5,
    )
