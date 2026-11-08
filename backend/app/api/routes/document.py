import json
from pathlib import PurePosixPath
from urllib.parse import quote
from uuid import UUID

from fastapi import HTTPException, APIRouter, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlmodel import select

from app.api.deps import SessionDep, OptionalUserDep
from app.auth.chat_access import user_can_view_chat, user_can_use_engine
from app.models import ChatMessage
from app.repositories import document_repo, chat_repo
from app.file_storage import get_file_storage
from app.utils.public_chat import chat_hides_sources

router = APIRouter()


@router.get("/documents/{doc_id}/download")
def download_file(
    request: Request,
    doc_id: int,
    session: SessionDep,
    user: OptionalUserDep,
    chat_id: UUID | None = None,
):
    if not (user and user.is_superuser):
        if chat_id is None:
            raise HTTPException(status_code=404, detail="File not found")
        chat = chat_repo.must_get(session, chat_id)
        if (
            not user_can_view_chat(chat, user, request.state.browser_id)
            or not user_can_use_engine(chat.engine, user)
            or chat_hides_sources(chat)
        ):
            raise HTTPException(status_code=404, detail="File not found")
        # A public engine does not publish its whole knowledge base. Only a
        # document cited by a chat this visitor may read can be downloaded.
        citation = session.exec(
            select(ChatMessage.id)
            .where(
                ChatMessage.chat_id == chat_id,
                func.json_contains(ChatMessage.sources, json.dumps({"id": doc_id})),
            )
            .limit(1)
        ).first()
        if citation is None:
            raise HTTPException(status_code=404, detail="File not found")
    doc = document_repo.must_get(session, doc_id)

    name = doc.source_uri
    if not name.startswith("uploads/"):
        raise HTTPException(status_code=404, detail="File not found")
    filestorage = get_file_storage()
    if filestorage.exists(name):
        file_size = filestorage.size(name)
        filename = quote(PurePosixPath(name).name, safe="")
        headers = {
            "Content-Length": str(file_size),
            "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
        }

        def iterfile():
            with filestorage.open(name) as f:
                while chunk := f.read(64 * 1024):
                    yield chunk

        return StreamingResponse(iterfile(), media_type=doc.mime_type, headers=headers)
    else:
        raise HTTPException(status_code=404, detail="File not found")
