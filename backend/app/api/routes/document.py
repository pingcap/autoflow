from fastapi import HTTPException, APIRouter
from fastapi.responses import StreamingResponse
from app.api.deps import SessionDep
from app.repositories import document_repo
from app.file_storage import get_file_storage

router = APIRouter()


@router.get("/documents/{doc_id}/download")
def download_file(doc_id: int, session: SessionDep):
    doc = document_repo.must_get(session, doc_id)

    name = doc.source_uri
    filestorage = get_file_storage()
    if filestorage.exists(name):
        file_size = filestorage.size(name)
        headers = {"Content-Length": str(file_size)}

        def iterfile():
            with filestorage.open(name) as f:
                yield from f

        return StreamingResponse(iterfile(), media_type=doc.mime_type, headers=headers)
    else:
        raise HTTPException(status_code=404, detail="File not found")
