from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import FileResponse

from sqlmodel import select, Session, col
from app.api.deps import SessionDep
from app.models import Document
from app.repositories import document_repo
import os

router = APIRouter()

@router.get("/documents/{doc_id}/download")
def download_file(
    doc_id: int,
    session: SessionDep
):
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code = 404, detail = "File not found")
    
    DATA_PATH = "../data"
    source_uri = os.path.join(DATA_PATH, doc.source_uri) 
    if os.path.exists(source_uri):
        return FileResponse(path = source_uri, filename = doc.name, media_type = doc.mime_type)
    else:
        raise HTTPException(status_code = 404, detail = "File not found")


